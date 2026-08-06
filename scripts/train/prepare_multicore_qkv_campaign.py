#!/usr/bin/env python3
"""Materialize and lock the ten-core adjacent-normal QKV campaign.

The protected selection manifest is the only source of core/FOV routing.  This
script prepares one immutable source artifact per opaque core alias, refits the
declared full-core transforms, materializes exact k=1,000 and k=5,000 graph
identities, writes 30 fully resolved production configs, registers the ten
dataset/split identities, and emits a GPU-balanced enqueue plan.  It does not
enqueue training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.adjacent_normal_selection import (  # noqa: E402
    load_adjacent_normal_route,
)
from spatial_benchmark.artifacts import (  # noqa: E402
    load_prepared_artifact,
    prepare_artifact,
)
from spatial_benchmark.configuration import (  # noqa: E402
    compose_config,
    validate_experiment_config,
)
from spatial_benchmark.full_core import (  # noqa: E402
    build_exact_mutual_knn_graph,
    load_and_refit_full_core,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


CAMPAIGN_ID = "cmp_20260728_adjacent_normal_10core_qkv_large_k"
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
RADIUS_GUARD_UM = 2000.0
GRAPH_K_VALUES = (1000, 5000)
MODEL_PARAMETER_COUNT = 36_749_480
PREPARATION_VERSION = "adjacent_normal_full_core_fit_v1"
EXPERIMENTAL_UNIT = "single_adjacent_normal_spatial_core"
PRODUCTION_EPOCHS = 300
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "a460de90ba9998e6817cdf3fbbcd2416eec048ec3fe6c825db8677b70b8b297f"
)

_BASE_CONFIGS = {
    "k1000": "configs/experiment/full_core_qkv_k1000.yaml",
    "k5000": "configs/experiment/full_core_qkv_k5000.yaml",
    "matched_self": "configs/experiment/full_core_qkv_matched_self.yaml",
}


class MulticoreCampaignPreparationError(ValueError):
    """Raised when a protected input or locked campaign identity is invalid."""


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            dict(payload),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _project_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_PROJECT_ROOT))
    except ValueError as exc:
        raise MulticoreCampaignPreparationError(
            "Campaign artifacts must remain under the project root."
        ) from exc


def _preparation_config(
    *,
    selection_manifest: Path,
    alias: str,
) -> dict[str, Any]:
    return {
        "version": 1,
        "project_root": str(_PROJECT_ROOT),
        "data": {
            "raw_dir": "data/raw",
            "legacy_workbook": "data/clinical/Gastric Study_Old.xlsx",
            "pathology_review_workbook": "data/clinical/Gastric Study.xlsx",
            "core_map_csv": "data/clinical/fov_core_map.csv",
            "chunksize": 8192,
            "expected_biological_probes": 1000,
            "pixel_size_um": 0.120281,
            "qc_policy": "all",
            "selection_source": "protected_adjacent_normal_manifest",
            "selection_manifest": _project_reference(selection_manifest),
            "selection_alias": alias,
        },
        "split": {
            "block_size_um": 300.0,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 20260728,
            "fov_aware": True,
        },
        "preprocessing": {
            "log1p_metadata": True,
            "epsilon": 1e-8,
        },
        "graph": {
            "k": 12,
            "radius_um": 50.0,
            "symmetry": "union",
            "min_distance_um": 0.0,
            "rbf_bins": 8,
            "edge_dropout": 0.1,
        },
        "masking": {
            "mask_seed": 20260728,
            "curriculum": "P+N+B",
            "warmup_epochs": 10,
            "partial_gene_rate": 0.2,
            "whole_node_rate": 0.1,
            "block_node_rate": 0.1,
            "block_width_um": None,
            "block_shape": "disk",
            "validation_replicates": 3,
            "test_replicates": 5,
        },
    }


def _split_identity(
    *,
    dataset_fingerprint: str,
    n_nodes: int,
) -> tuple[dict[str, Any], str, str]:
    basis = {
        "schema": "full_core_no_holdout_roles_v1",
        "dataset_fingerprint": dataset_fingerprint,
        "n_nodes": n_nodes,
        "role_assignment": (
            "every row in verified prepared artifact order assigned fit"
        ),
        "role_counts": {
            "fit": n_nodes,
            "validation": 0,
            "test": 0,
        },
        "experimental_unit": EXPERIMENTAL_UNIT,
    }
    fingerprint = canonical_sha256(basis)
    return basis, fingerprint, fingerprint[:16]


def _dataset_config(
    *,
    alias: str,
    prepared_artifact: Path,
    core: Any,
) -> dict[str, Any]:
    fingerprint = core.checksums.preprocessing_sha256
    split_basis, split_fingerprint, split_id = _split_identity(
        dataset_fingerprint=fingerprint,
        n_nodes=core.n_nodes,
    )
    alias_key = alias.lower().replace("-", "")
    return {
        "dataset_id": f"cosmx_{alias_key}_adjacent_normal_full_core_fit_v1",
        "version": PREPARATION_VERSION,
        "split_id": split_id,
        "dataset_fingerprint": fingerprint,
        "dataset_fingerprint_role": (
            "materialized_full_core_preprocessing_checksum"
        ),
        "dataset_fingerprint_basis": {
            "recipe_schema": "full_core_fit_v1",
            "source_dataset_id": "cosmx_gastric_snapshot_20260724",
            "source_version": "26040302SO_1+26040302SO_2",
            "source_artifact_id": core.checksums.source_artifact_id,
            "source_prepared_data_sha256": (
                core.checksums.source_prepared_data_sha256
            ),
            "biological_targets": core.n_genes,
            "technical_control_prefixes_excluded": [
                "Negative",
                "SystemControl",
            ],
            "expression_transform": "gene_wise_log1p_standardization",
            "metadata_transform": (
                "inverse_prepared_transform_then_median_imputation_"
                "log1p_standardization"
            ),
            "numerical_epsilon": 1e-8,
            "fit_scope": f"all_{core.n_nodes}_nodes_transductive",
            "materialized_preprocessing_sha256": fingerprint,
        },
        "split_fingerprint": split_fingerprint,
        "split_fingerprint_status": (
            "verified_materialized_no_holdout_role_assignment"
        ),
        "split_fingerprint_basis": split_basis,
        "preprocessing_version": PREPARATION_VERSION,
        "prepared_artifact_reference": _project_reference(
            prepared_artifact
        ),
        "task": "masked_expression_regression",
        "target_scale": (
            "full_core_fitted_gene_wise_standardized_log1p_counts"
        ),
        "biological_target_count": core.n_genes,
        "technical_control_prefixes_excluded": [
            "Negative",
            "SystemControl",
        ],
        "preprocessing_fit_scope": "all_nodes_transductive",
        "experimental_unit": EXPERIMENTAL_UNIT,
        "biological_unit_alias": alias,
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "generalization_scope": "held_in_reconstruction_only",
        "validation_or_test_partition_present": False,
        "patient_generalization_supported": False,
    }


def _graph_record(graph: Any) -> dict[str, Any]:
    return {
        "k": graph.k,
        "n_directed_edges": graph.qc.n_directed_edges,
        "graph_sha256": graph.checksums.graph_sha256,
        "candidate_kth_distance_max_um": (
            graph.qc.candidate_kth_distance_max_um
        ),
        "radius_guard_margin_um": graph.qc.radius_guard_margin_um,
        "mean_degree": graph.qc.mean_degree,
        "n_components": graph.qc.n_components,
        "n_isolated_nodes": graph.qc.n_isolated_nodes,
    }


def _locked_graph_section(
    base: Mapping[str, Any],
    *,
    graph_record: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    k = int(graph_record["k"])
    result.update(
        {
            "neighbor_k": k,
            "k": k,
            "radius_um": RADIUS_GUARD_UM,
            "radius_guard_um": RADIUS_GUARD_UM,
            "radius_role": "common_nontruncating_post_knn_guard",
            "candidate_selection": "exact_knn_before_radius_guard_validation",
            "expected_materialized_graph_sha256": str(
                graph_record["graph_sha256"]
            ),
            "expected_directed_edges": int(
                graph_record["n_directed_edges"]
            ),
            "construction_workers": 8,
        }
    )
    return result


def _production_config(
    *,
    alias: str,
    arm: str,
    dataset: Mapping[str, Any],
    graph_records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    source = _PROJECT_ROOT / _BASE_CONFIGS[arm]
    config = compose_config(source)
    config["dataset"] = deepcopy(dict(dataset))
    graph_k = 1000 if arm == "k1000" else 5000
    config["graph"] = _locked_graph_section(
        config["graph"],
        graph_record=graph_records[graph_k],
    )
    alias_key = alias.lower().replace("-", "")
    config["campaign"] = {
        "campaign_id": CAMPAIGN_ID,
        "display_name": (
            "Ten-core adjacent-normal QKV-GAT large-k replication"
        ),
    }
    config["experiment"] = {
        "variant_label": f"{alias_key}_qkv_{arm}_full_core",
        "biological_unit_alias": alias,
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "estimand": "held_in_full_core_whole_node_masked_reconstruction",
        "permitted_claim": (
            "ten_core_adjacent_normal_transductive_representation_capacity"
        ),
        "paired_within_core": True,
    }
    config["classification"] = {
        "schema_version": 1,
        "lifecycle_stage": "exploratory_screen",
        "study_axis": "adjacent_normal_10core_qkv_large_k_capacity",
        "retention_class": "retain_exploratory_evidence",
        "classification_confidence": "high",
    }
    config["seed"] = 0
    config["fold"] = 0
    config["attempt"] = 1
    if int(config["trainer"]["max_epochs"]) != PRODUCTION_EPOCHS:
        raise MulticoreCampaignPreparationError(
            "The production base config no longer declares 300 epochs."
        )
    validate_experiment_config(config)
    return config


def _register_dataset_and_split(
    registry: Registry,
    *,
    dataset: Mapping[str, Any],
    n_nodes: int,
) -> None:
    prepared = str(dataset["prepared_artifact_reference"])
    registry.register_dataset(
        str(dataset["dataset_id"]),
        str(dataset["version"]),
        display_name=(
            "Protected pathology-confirmed adjacent-normal full-core fit "
            f"({dataset['biological_unit_alias']})"
        ),
        protected_source_path=prepared,
        raw_fingerprint=str(
            dataset["dataset_fingerprint_basis"][
                "source_prepared_data_sha256"
            ]
        ),
        preprocessing_version=str(dataset["preprocessing_version"]),
        processed_fingerprint=str(dataset["dataset_fingerprint"]),
        aggregate_sample_count=n_nodes,
        graph_count=2,
        node_feature_schema=(
            "1000 biological probes plus 22 permitted morphology/imaging "
            "measurements; no identifiers or coordinates as node covariates"
        ),
        edge_feature_schema="17 standardized spatial geometry attributes",
        creation_date="2026-07-28",
        status="materialized_and_verified",
        verification_status="verified_materialized_full_core_preprocessing",
        metadata={
            "biological_unit_alias": dataset["biological_unit_alias"],
            "tissue_context": dataset["tissue_context"],
            "validation_or_test_partition_present": False,
        },
    )
    registry.register_split(
        str(dataset["split_id"]),
        dataset_id=str(dataset["dataset_id"]),
        dataset_version=str(dataset["version"]),
        method="deterministic_all_cells_full_core_fit_no_holdout",
        unit=EXPERIMENTAL_UNIT,
        seed=None,
        fold_count=1,
        stratification=[],
        fingerprint=str(dataset["split_fingerprint"]),
        protected_path=prepared,
        verification_status="verified_all_materialized_nodes_assigned_fit",
        metadata={
            "role_counts": dataset["split_fingerprint_basis"][
                "role_counts"
            ],
            "generalization_estimate": False,
        },
    )


def _assign_gpus(
    jobs: list[dict[str, Any]],
    *,
    gpu_count: int = 8,
) -> list[dict[str, Any]]:
    loads = [0.0] * gpu_count
    ordered = sorted(
        jobs,
        key=lambda item: (
            -float(item["estimated_work"]),
            str(item["alias"]),
            str(item["arm"]),
        ),
    )
    assigned: list[dict[str, Any]] = []
    for job in ordered:
        gpu = min(range(gpu_count), key=lambda index: (loads[index], index))
        record = {**job, "requested_gpu": gpu}
        assigned.append(record)
        loads[gpu] += float(job["estimated_work"])
    return assigned


def prepare_campaign(
    *,
    selection_manifest: Path,
    protected_root: Path,
    state_dir: Path,
    database: Path,
) -> dict[str, Any]:
    observed_selection_sha256 = sha256_file(selection_manifest)
    if observed_selection_sha256 != EXPECTED_SELECTION_MANIFEST_SHA256:
        raise MulticoreCampaignPreparationError(
            "Protected selection manifest differs from the locked external "
            "SHA-256."
        )
    routes = {
        alias: load_adjacent_normal_route(selection_manifest, alias)
        for alias in ALIASES
    }
    if tuple(routes) != ALIASES:
        raise MulticoreCampaignPreparationError(
            "Protected selection aliases do not match the locked ten-core set."
        )

    preparation_config_dir = state_dir / "preparation_configs"
    production_config_dir = state_dir / "production_configs"
    materialized: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    registry = Registry(database)
    campaign_plan_path = (
        _PROJECT_ROOT
        / "experiments"
        / "campaigns"
        / CAMPAIGN_ID
        / "campaign.yaml"
    )
    campaign_plan = yaml.safe_load(
        campaign_plan_path.read_text(encoding="utf-8")
    )
    registry.create_campaign(
        CAMPAIGN_ID,
        name="Ten-core adjacent-normal QKV-GAT large-k replication",
        scientific_question=(
            "Across ten independent pathology-confirmed adjacent-normal "
            "cores, does exact QKV graph attention at k=5000 improve "
            "held-in masked-expression reconstruction over k=1000 and a "
            "parameter-matched cell-only control?"
        ),
        config=campaign_plan,
        status="pilot",
    )

    for alias in ALIASES:
        preparation_config_path = (
            preparation_config_dir / f"{alias.lower()}_prepare.yaml"
        )
        _write_yaml(
            preparation_config_path,
            _preparation_config(
                selection_manifest=selection_manifest,
                alias=alias,
            ),
        )
        prepared_artifact = protected_root / alias.lower() / "prepared_v1"
        if not prepared_artifact.exists():
            prepare_artifact(
                preparation_config_path,
                prepared_artifact,
                command=[
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--selection-manifest",
                    _project_reference(selection_manifest),
                ],
            )
        source_manifest, _, _ = load_prepared_artifact(
            prepared_artifact,
            load_arrays=False,
        )
        source_selection = source_manifest.get("selection")
        source_inputs = source_manifest.get("inputs")
        if (
            not isinstance(source_selection, Mapping)
            or source_selection.get("opaque_alias") != alias
            or not isinstance(source_inputs, Mapping)
            or source_inputs.get(
                "protected_selection_manifest", {}
            ).get("sha256")
            != EXPECTED_SELECTION_MANIFEST_SHA256
        ):
            raise MulticoreCampaignPreparationError(
                "Existing prepared artifact does not match its locked alias "
                "and protected selection manifest."
            )

        core = load_and_refit_full_core(prepared_artifact)
        if core.n_nodes <= max(GRAPH_K_VALUES):
            raise MulticoreCampaignPreparationError(
                f"{alias} does not contain enough cells for literal k=5,000."
            )
        dataset = _dataset_config(
            alias=alias,
            prepared_artifact=prepared_artifact,
            core=core,
        )

        graph_records: dict[int, dict[str, Any]] = {}
        for k in GRAPH_K_VALUES:
            graph = build_exact_mutual_knn_graph(
                core.coordinates_um,
                k=k,
                radius_guard_um=RADIUS_GUARD_UM,
                query_chunk_size=512,
                receiver_chunk_size=64,
                mutual_search_chunk_size=4_000_000,
                workers=8,
                epsilon=1e-8,
            )
            graph_records[k] = _graph_record(graph)
            del graph
            gc.collect()

        _register_dataset_and_split(
            registry,
            dataset=dataset,
            n_nodes=core.n_nodes,
        )

        arm_configs: dict[str, str] = {}
        for arm in ("k5000", "k1000", "matched_self"):
            config = _production_config(
                alias=alias,
                arm=arm,
                dataset=dataset,
                graph_records=graph_records,
            )
            config_path = (
                production_config_dir
                / f"{alias.lower()}_{arm}.yaml"
            )
            _write_yaml(config_path, config)
            arm_configs[arm] = _project_reference(config_path)
            graph_k = 1000 if arm == "k1000" else 5000
            edge_work = float(
                graph_records[graph_k]["n_directed_edges"]
            )
            if arm == "matched_self":
                # Self training is node-local, but the locked runner still
                # rebuilds and verifies the paired k=5,000 graph.  Price both
                # the observed node-local training scale and a conservative
                # graph-construction fraction for GPU/CPU load balancing.
                edge_work = float(
                    core.n_nodes * 100
                    + 0.01
                    * graph_records[5000]["n_directed_edges"]
                )
            jobs.append(
                {
                    "alias": alias,
                    "arm": arm,
                    "config": _project_reference(config_path),
                    "estimated_work": edge_work,
                }
            )

        materialized.append(
            {
                "alias": alias,
                "prepared_artifact": _project_reference(prepared_artifact),
                "n_nodes": core.n_nodes,
                "n_genes": core.n_genes,
                "preprocessing_sha256": (
                    core.checksums.preprocessing_sha256
                ),
                "split_id": dataset["split_id"],
                "split_fingerprint": dataset["split_fingerprint"],
                "graphs": {
                    str(k): graph_records[k] for k in GRAPH_K_VALUES
                },
                "configs": arm_configs,
            }
        )
        del core
        gc.collect()

    assigned_jobs = _assign_gpus(jobs)
    for job in assigned_jobs:
        config_path = _PROJECT_ROOT / str(job["config"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["launcher"]["requested_gpu"] = str(job["requested_gpu"])
        validate_experiment_config(config)
        _write_yaml(config_path, config)
        job["config_sha256"] = canonical_sha256(config)
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "cohort": {
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "core_count": 10,
            "donor_count": 10,
            "slide_balance": {"SO_1": 5, "SO_2": 5},
            "true_normal_core_count": 0,
            "protected_selection_manifest_sha256": (
                observed_selection_sha256
            ),
        },
        "model": {
            "parameter_count": MODEL_PARAMETER_COUNT,
            "hidden_dim": 576,
            "graph_layers": 8,
            "attention_heads": 9,
            "attention_head_dim": 64,
            "fixed_epochs": PRODUCTION_EPOCHS,
        },
        "materialized_cores": materialized,
        "jobs": assigned_jobs,
        "job_count": len(assigned_jobs),
        "checksum": "",
    }
    result["checksum"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "checksum"}
    )
    _write_json(state_dir / "campaign_materialization.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument(
        "--protected-root",
        type=Path,
        default=(
            paths.data_root
            / "processed"
            / "adjacent_normal_10core_qkv_large_k_v1"
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=(
            paths.scratch_root
            / "locked_campaigns"
            / CAMPAIGN_ID
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking" / "bagm.sqlite3",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_campaign(
        selection_manifest=args.selection_manifest.resolve(),
        protected_root=args.protected_root.resolve(),
        state_dir=args.state_dir.resolve(),
        database=args.database.resolve(),
    )
    print(
        json.dumps(
            {
                "campaign_id": result["campaign_id"],
                "core_count": result["cohort"]["core_count"],
                "job_count": result["job_count"],
                "parameter_count": result["model"]["parameter_count"],
                "status": "materialized_not_enqueued",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

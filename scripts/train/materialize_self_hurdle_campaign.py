#!/usr/bin/env python3
"""Materialize two graphless self-hurdle pilots and two science configs.

This command is preparation-only. It verifies the frozen contract and the
checksum-bound source materialization, then atomically writes resolved configs
and a receipt. It does not register, enqueue, or train anything.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.self_hurdle import (  # noqa: E402
    SelfHurdleModel,
    trainable_parameter_count,
)


CAMPAIGN_ID = "cmp_20260729_self_hurdle_full_core_capacity"
SOURCE_CAMPAIGN_ID = "cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
CONTRACT_RELATIVE = Path(
    "experiments/campaigns"
) / CAMPAIGN_ID / "frozen_task_contract.yaml"
CONTRACT_SHA256 = (
    "26cf4f094d843c4fa9020c52e1d45988f8e4a0c43f20beb186e4742b2dacba00"
)
SOURCE_RECEIPT_RELATIVE = (
    Path("scratch/locked_campaigns")
    / SOURCE_CAMPAIGN_ID
    / "locked_config_materialization.json"
)
LOCKED_RELATIVE = Path("scratch/locked_campaigns") / CAMPAIGN_ID
RECEIPT_NAME = "locked_config_materialization.json"
RECEIPT_KIND = "self_hurdle_locked_config_materialization_v1"
RESOURCE_ALIASES = ("ANC-03", "ANC-05")
SCIENCE_ALIASES = RESOURCE_ALIASES
RESOURCE_GPUS = {"ANC-03": 0, "ANC-05": 1}
SCIENCE_GPUS = {"ANC-03": 2, "ANC-05": 3}
SAFE_GPUS = (0, 1, 2, 3, 5, 6, 7)
EXPECTED_PARAMETER_COUNT = 16_917_200
MODEL_COMPONENT = "configs/model/self_hurdle_large.yaml"
GRAPH_COMPONENT = "configs/graph/self_only_disabled.yaml"
TRAINER_COMPONENTS = {
    "resource": "configs/trainer/self_hurdle_resource_pilot_2.yaml",
    "science": "configs/trainer/self_hurdle_fixed_200.yaml",
}
EVALUATION_COMPONENTS = {
    "resource": (
        "configs/evaluation/held_in_self_hurdle_resource_pilot_v1.yaml"
    ),
    "science": "configs/evaluation/held_in_self_hurdle_v1.yaml",
}
LAUNCHER_COMPONENT = "configs/launcher/local_single_gpu_3090.yaml"


class SelfHurdleMaterializationError(RuntimeError):
    pass


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelfHurdleMaterializationError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise SelfHurdleMaterializationError(
            f"non-finite JSON constant {value!r}"
        )

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelfHurdleMaterializationError(
                    f"duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject,
            object_pairs_hook=unique,
        )
    except SelfHurdleMaterializationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelfHurdleMaterializationError(
            f"invalid source receipt: {path}"
        ) from exc
    return dict(_mapping(value, "source receipt"))


def _verify_checksum(value: Mapping[str, Any], label: str) -> str:
    observed = value.get("checksum")
    if not isinstance(observed, str) or len(observed) != 64:
        raise SelfHurdleMaterializationError(f"{label} checksum is malformed")
    payload = dict(value)
    payload.pop("checksum", None)
    if canonical_sha256(payload) != observed:
        raise SelfHurdleMaterializationError(
            f"{label} checksum does not verify"
        )
    return observed


def _component(reference: str) -> tuple[dict[str, Any], str]:
    path = (PROJECT_ROOT / reference).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SelfHurdleMaterializationError(
            "component escapes project root"
        ) from exc
    if not path.is_file():
        raise SelfHurdleMaterializationError(
            f"component is missing: {reference}"
        )
    value = dict(load_yaml_mapping(path))
    if len(value) != 1:
        raise SelfHurdleMaterializationError(
            f"component must have one namespaced group: {reference}"
        )
    return deepcopy(next(iter(value.values()))), sha256_file(path)


def _source_materialization() -> dict[str, Any]:
    path = PROJECT_ROOT / SOURCE_RECEIPT_RELATIVE
    source = _strict_json(path)
    _verify_checksum(source, "source materialization")
    if (
        source.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or source.get("training_performed") is not False
        or source.get("registry_mutation_performed") is not False
    ):
        raise SelfHurdleMaterializationError(
            "source materialization metadata is incompatible"
        )
    return source


def _source_core_map(
    source: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    values = source.get("cores")
    if not isinstance(values, list):
        raise SelfHurdleMaterializationError("source cores must be a list")
    result = {
        str(_mapping(item, "source core").get("alias")): _mapping(
            item, "source core"
        )
        for item in values
    }
    if any(alias not in result for alias in RESOURCE_ALIASES):
        raise SelfHurdleMaterializationError("required source core is missing")
    return result


def _source_config(alias: str, source: Mapping[str, Any]) -> dict[str, Any]:
    jobs = source.get("production_jobs")
    if not isinstance(jobs, list):
        raise SelfHurdleMaterializationError(
            "source production jobs must be a list"
        )
    matches = [
        _mapping(item, "source production job")
        for item in jobs
        if _mapping(item, "source production job").get("alias") == alias
        and _mapping(item, "source production job").get("arm")
        == "hybrid-matched-self"
    ]
    if len(matches) != 1:
        raise SelfHurdleMaterializationError(
            f"source matched-self config is not unique for {alias}"
        )
    job = matches[0]
    reference = Path(str(job.get("config")))
    if reference.is_absolute():
        raise SelfHurdleMaterializationError("source config must be relative")
    path = (PROJECT_ROOT / reference).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SelfHurdleMaterializationError(
            "source config escapes project root"
        ) from exc
    if (
        not path.is_file()
        or sha256_file(path) != job.get("file_sha256")
    ):
        raise SelfHurdleMaterializationError(
            f"source config checksum drifted for {alias}"
        )
    config = dict(load_yaml_mapping(path))
    validate_experiment_config(config)
    return config


def _parameter_audit(model_config: Mapping[str, Any]) -> dict[str, Any]:
    model = SelfHurdleModel(
        num_genes=1000,
        expression_mean=torch.zeros(1000),
        expression_scale=torch.ones(1000),
        node_covariate_dim=22,
        hidden_dim=int(model_config["hidden_dim"]),
        decoder_dim=int(model_config["decoder_dim"]),
        ffn_dim=int(model_config["ffn_dim"]),
        residual_blocks=int(model_config["residual_blocks"]),
        dropout=float(model_config["dropout"]),
    )
    count = trainable_parameter_count(model)
    shapes = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if count != EXPECTED_PARAMETER_COUNT:
        raise SelfHurdleMaterializationError(
            f"parameter count changed: {count}"
        )
    return {
        "trainable_parameter_count": count,
        "named_parameter_shapes_sha256": canonical_sha256(shapes),
        "uses_graph_inputs": False,
        "uses_edge_inputs": False,
    }


def _count_representation() -> dict[str, Any]:
    return {
        "schema": "hurdle_detection_plus_positive_standardized_log1p_v1",
        "source_scale": "raw_biological_probe_counts",
        "input_states": {
            "schema": (
                "hybrid_raw_count_0_1_2_3_4_7_8_15_16_31_32plus_v1"
            ),
            "num_states": 8,
            "mask_token_id": 8,
            "mask_token_is_output": False,
        },
        "output_channels_per_gene": 2,
        "output_channels": [
            "detection_logit",
            "positive_standardized_log1p",
        ],
        "detection_threshold": 0.5,
        "continuous_transform": "per_gene_all_fit_standardized_log1p",
        "fit_required": False,
    }


def _config(
    *,
    alias: str,
    stage: str,
    source_config: Mapping[str, Any],
    model: Mapping[str, Any],
    graph: Mapping[str, Any],
    trainer: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    launcher: Mapping[str, Any],
    materialization_reference: str,
) -> dict[str, Any]:
    if stage not in {"resource", "science"}:
        raise SelfHurdleMaterializationError("invalid stage")
    gpu = (
        RESOURCE_GPUS[alias]
        if stage == "resource"
        else SCIENCE_GPUS[alias]
    )
    dataset = deepcopy(_mapping(source_config.get("dataset"), "dataset"))
    dataset.update(
        {
            "task": "masked_expression_hurdle_count",
            "target_scale": (
                "raw_biological_probe_counts_with_per_gene_all_fit_"
                "standardized_log1p"
            ),
            "count_representation": _count_representation(),
            "frozen_task_contract_sha256": CONTRACT_SHA256,
        }
    )
    features = deepcopy(_mapping(source_config.get("features"), "features"))
    features["use_edge_features"] = False
    features["edge_features"] = []
    expression = deepcopy(
        _mapping(features.get("node_expression"), "node expression")
    )
    expression.update(
        {
            "discrete_transform": "fixed_hybrid_count_states",
            "continuous_transform": (
                "per_gene_all_fit_standardized_log1p"
            ),
            "masked_discrete_value": "input_only_mask_token_8",
            "masked_continuous_value": 0.0,
            "explicit_mask_authoritative_inside_model": True,
        }
    )
    features["node_expression"] = expression
    selected_launcher = deepcopy(dict(launcher))
    selected_launcher.update(
        {
            "requested_gpu": str(gpu),
            "requested_gpu_count": 1,
            "disk_safety_max_used_decimal_gb": 55.0,
        }
    )
    selected_trainer = deepcopy(dict(trainer))
    if stage == "science":
        authorization = _mapping(
            selected_trainer.get("amp_authorization"),
            "science AMP authorization",
        )
        selected_trainer["amp_authorization"] = {
            **dict(authorization),
            "frozen_contract_sha256": CONTRACT_SHA256,
        }
    config = {
        "version": 1,
        "model": deepcopy(dict(model)),
        "masking": deepcopy(
            dict(_mapping(source_config.get("masking"), "masking"))
        ),
        "dataset": dataset,
        "features": features,
        "graph": deepcopy(dict(graph)),
        "trainer": selected_trainer,
        "evaluation": deepcopy(dict(evaluation)),
        "launcher": selected_launcher,
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "display_name": (
                "Self-only continuous-hurdle full-core capacity"
            ),
            "exploratory": True,
            "frozen_contract": CONTRACT_RELATIVE.as_posix(),
            "frozen_contract_sha256": CONTRACT_SHA256,
        },
        "experiment": {
            "variant_label": (
                f"{alias.lower().replace('-', '')}_self_hurdle_"
                f"{'resource_2ep' if stage == 'resource' else 'fixed_200ep'}"
            ),
            "arm": "self-hurdle",
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "estimand": (
                "held_in_full_core_whole_node_masked_raw_count_"
                "reconstruction"
            ),
            "permitted_claim": (
                "resource_feasibility_only"
                if stage == "resource"
                else "two_core_exploratory_within_cell_representation_capacity"
            ),
            "resource_pilot": stage == "resource",
            "conclusion_eligible": stage == "science",
            "graph_arms_authorized": False,
        },
        "classification": {
            "schema_version": 1,
            "lifecycle_stage": (
                "diagnostic"
                if stage == "resource"
                else "exploratory_screen"
            ),
            "study_axis": "self_hurdle_full_core_capacity",
            "scientific_variant": (
                f"self_hurdle_large_{alias.lower().replace('-', '')}"
            ),
            "retention_class": (
                "retain_diagnostic"
                if stage == "resource"
                else "retain_exploratory_evidence"
            ),
            "classification_confidence": "high",
        },
        "metadata": {
            "locked_config_materialization_receipt": (
                materialization_reference
            ),
            "frozen_scientific_contract": True,
            "execution_role": (
                "resource_pilot"
                if stage == "resource"
                else "science"
            ),
            "no_graph_construction_or_input": True,
            "science_requires_resource_gate": stage == "science",
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }
    validate_experiment_config(config)
    return config


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize(output_root: Path | None = None) -> dict[str, Any]:
    contract_path = PROJECT_ROOT / CONTRACT_RELATIVE
    if (
        not contract_path.is_file()
        or sha256_file(contract_path) != CONTRACT_SHA256
    ):
        raise SelfHurdleMaterializationError(
            "frozen task contract checksum drifted"
        )
    contract = load_yaml_mapping(contract_path)
    if (
        contract.get("campaign_id") != CAMPAIGN_ID
        or contract.get("frozen_before_new_training") is not True
        or contract.get("exploratory") is not True
    ):
        raise SelfHurdleMaterializationError(
            "frozen task contract metadata is invalid"
        )
    source = _source_materialization()
    source_cores = _source_core_map(source)

    component_references = [
        MODEL_COMPONENT,
        GRAPH_COMPONENT,
        *TRAINER_COMPONENTS.values(),
        *EVALUATION_COMPONENTS.values(),
        LAUNCHER_COMPONENT,
    ]
    loaded: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for reference in component_references:
        loaded[reference], hashes[reference] = _component(reference)
    parameter_audit = _parameter_audit(loaded[MODEL_COMPONENT])

    destination = (
        output_root.resolve()
        if output_root is not None
        else (PROJECT_ROOT / LOCKED_RELATIVE).resolve()
    )
    try:
        destination.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SelfHurdleMaterializationError(
            "output root must remain under project root"
        ) from exc
    if destination.exists():
        raise SelfHurdleMaterializationError(
            f"locked output already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{CAMPAIGN_ID}.",
            dir=destination.parent,
        )
    )
    try:
        materialization_reference = (
            destination.relative_to(PROJECT_ROOT).as_posix()
            + f"/{RECEIPT_NAME}"
        )
        jobs: dict[str, list[dict[str, Any]]] = {
            "resource": [],
            "science": [],
        }
        for stage, aliases in (
            ("resource", RESOURCE_ALIASES),
            ("science", SCIENCE_ALIASES),
        ):
            for alias in aliases:
                source_config = _source_config(alias, source)
                config = _config(
                    alias=alias,
                    stage=stage,
                    source_config=source_config,
                    model=loaded[MODEL_COMPONENT],
                    graph=loaded[GRAPH_COMPONENT],
                    trainer=loaded[TRAINER_COMPONENTS[stage]],
                    evaluation=loaded[EVALUATION_COMPONENTS[stage]],
                    launcher=loaded[LAUNCHER_COMPONENT],
                    materialization_reference=materialization_reference,
                )
                directory = (
                    "resource_configs"
                    if stage == "resource"
                    else "science_configs"
                )
                filename = (
                    f"{alias.lower()}_self_hurdle_"
                    f"{'resource_2ep' if stage == 'resource' else 'fixed_200ep'}"
                    ".yaml"
                )
                path = temporary / directory / filename
                _write_yaml(path, config)
                relative = (
                    destination.relative_to(PROJECT_ROOT)
                    / directory
                    / filename
                ).as_posix()
                jobs[stage].append(
                    {
                        "alias": alias,
                        "arm": "self-hurdle",
                        "stage": stage,
                        "config": relative,
                        "config_sha256": canonical_sha256(config),
                        "file_sha256": sha256_file(path),
                        "requested_gpu": (
                            RESOURCE_GPUS[alias]
                            if stage == "resource"
                            else SCIENCE_GPUS[alias]
                        ),
                        "n_nodes": int(source_cores[alias]["n_nodes"]),
                        "n_genes": int(source_cores[alias]["n_genes"]),
                    }
                )
        core_receipts = [
            {
                "alias": alias,
                "prepared_artifact": source_cores[alias][
                    "prepared_artifact"
                ],
                "n_nodes": int(source_cores[alias]["n_nodes"]),
                "n_genes": int(source_cores[alias]["n_genes"]),
                "preprocessing_sha256": source_cores[alias][
                    "preprocessing_sha256"
                ],
                "source_prepared_data_sha256": source_cores[alias][
                    "source_prepared_data_sha256"
                ],
                "split_fingerprint": source_cores[alias][
                    "split_fingerprint"
                ],
                "split_id": source_cores[alias]["split_id"],
            }
            for alias in RESOURCE_ALIASES
        ]
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "receipt_kind": RECEIPT_KIND,
            "campaign_id": CAMPAIGN_ID,
            "frozen_contract": {
                "reference": CONTRACT_RELATIVE.as_posix(),
                "sha256": CONTRACT_SHA256,
            },
            "source_materialization": {
                "reference": SOURCE_RECEIPT_RELATIVE.as_posix(),
                "checksum": source["checksum"],
            },
            "component_file_sha256": hashes,
            "parameter_audit": parameter_audit,
            "graph_contract": {
                "construction_performed": False,
                "model_graph_inputs": False,
                "model_edge_inputs": False,
                "expected_directed_edges": 0,
            },
            "resource_limits": {
                "pilot_peak_vram_gib_maximum": 12.0,
                "science_peak_vram_gib_maximum": 20.5,
                "fp32_amp_loss_discrepancy_maximum": 0.001,
                "projected_gpu_hours_per_science_run_maximum": 6.0,
                "projected_aggregate_science_gpu_hours_maximum": 12.0,
                "absolute_aggregate_gpu_hours_maximum": 24.0,
                "filesystem_used_decimal_gb_hard_stop": 55.0,
            },
            "allowed_gpu_ids": list(SAFE_GPUS),
            "cores": core_receipts,
            "resource_jobs": jobs["resource"],
            "science_jobs": jobs["science"],
            "counts": {
                "cores": 2,
                "resource_configs": 2,
                "science_configs": 2,
            },
            "resource_gate_receipt_reference": (
                destination.relative_to(PROJECT_ROOT)
                / "resource_gate_receipt.json"
            ).as_posix(),
            "registry_mutation_performed": False,
            "queue_mutation_performed": False,
            "training_performed": False,
        }
        receipt["checksum"] = canonical_sha256(receipt)
        _write_json(temporary / RECEIPT_NAME, receipt)
        temporary.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = materialize(args.output_root)
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "resource_configs": len(receipt["resource_jobs"]),
                "science_configs": len(receipt["science_jobs"]),
                "checksum": receipt["checksum"],
                "registry_mutation_performed": False,
                "queue_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


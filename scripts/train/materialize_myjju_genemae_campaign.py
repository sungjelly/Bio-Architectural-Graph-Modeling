#!/usr/bin/env python3
"""Materialize checksum-bound MyJJu GeneMAE pilot and production configs.

This entry point is preparation-only.  It verifies the frozen task contract,
external architecture source, ten-core cohort, source-style tiled graphs, and
the exact BAGM partial-gene mask identities before atomically writing one pilot
and seven production configurations.  It never registers or enqueues a run.

The ``verify-pilot`` subcommand can subsequently bind a successful registered
pilot artifact to a production gate receipt.  It also performs no registry or
queue mutation.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.pooled_full_core import (  # noqa: E402
    ANC_ALIASES,
    EXPECTED_N_GENES,
    EXPECTED_TOTAL_NODES,
    load_pooled_full_core_cohort,
)
from spatial_benchmark.queueing import command_for_config  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


# Bootstrap discovery is script-relative, but all reusable project paths honor
# BAGM_ROOT and the canonical path resolver after imports are available.
PROJECT_ROOT = current_paths().project_root


CAMPAIGN_ID = "cmp_20260730_myjju_genemae_10core_comparison"
COMPARATOR_CAMPAIGN_ID = (
    "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
)
CONTRACT_SHA256 = (
    "6f171b5bece63df943dd8fe679121fbdd41baf290339d64576dc6bfec847eada"
)
SOURCE_AUDIT_SHA256 = (
    "6f4eb7a557d5b488fe84d7b4f0d26d07f97dd6349f9cdd3203b12a426d6def91"
)
EXPECTED_PARAMETER_COUNT = 6_888_016
MODEL_SEEDS = tuple(range(7))
SAFE_GPU_IDS = (0, 1, 2, 3, 5, 6, 7)
SEED_GPU_MAP = dict(zip(MODEL_SEEDS, SAFE_GPU_IDS, strict=True))
PILOT_GPU_ID = 0
CONTRACT_RELATIVE = (
    Path("experiments/campaigns") / CAMPAIGN_ID / "frozen_task_contract.yaml"
)
CAMPAIGN_RELATIVE = Path("experiments/campaigns") / CAMPAIGN_ID / "campaign.yaml"
SOURCE_AUDIT_RELATIVE = (
    Path("experiments/campaigns") / CAMPAIGN_ID / "external_source_audit.yaml"
)
LOCKED_RELATIVE = Path("scratch/locked_campaigns") / CAMPAIGN_ID
RECEIPT_NAME = "locked_config_materialization.json"
PILOT_GATE_NAME = "pilot_gate_receipt.json"
RECEIPT_KIND = "myjju_genemae_locked_config_materialization_v1"
PILOT_GATE_KIND = "myjju_genemae_resource_pilot_gate_v1"
COMPARATOR_RECEIPT_RELATIVE = (
    Path("scratch/locked_campaigns")
    / COMPARATOR_CAMPAIGN_ID
    / "locked_config_materialization.json"
)
COMPARATOR_RECEIPT_FILE_SHA256 = (
    "723ada09229530cf7f3b8f05e031173b4cc316c50324b041b56a03213c76aba1"
)
COMPARATOR_RECEIPT_CHECKSUM = (
    "d1a49b0428b1f90ec17f84581264fe4dc4ac3eca56526a0e29bcc761af02fdaa"
)
COMPONENTS = {
    "model": "configs/model/myjju_genemae_source_fidelity.yaml",
    "features": "configs/features/myjju_genemae_expression_only.yaml",
    "graph": "configs/graph/myjju_genemae_k15_tiled_union.yaml",
    "launcher": "configs/launcher/local_single_gpu_3090.yaml",
}
TRAINERS = {
    "pilot": "configs/trainer/myjju_genemae_resource_pilot_2.yaml",
    "production": "configs/trainer/myjju_genemae_fixed_200.yaml",
}
EVALUATIONS = {
    "pilot": (
        "configs/evaluation/"
        "held_in_pooled_10core_myjju_genemae_pilot_v1.yaml"
    ),
    "production": (
        "configs/evaluation/held_in_pooled_10core_myjju_genemae_v1.yaml"
    ),
}
DATASET_COMPONENT = "configs/dataset/adjacent_normal_10core_pooled_fit_v1.yaml"
IMPLEMENTATION_FILES = {
    "model_module": "src/spatial_benchmark/myjju_genemae.py",
    "runner": "scripts/train/run_myjju_genemae_pooled.py",
    "materializer": "scripts/train/materialize_myjju_genemae_campaign.py",
}
PINNED_MODEL_MODULE_SHA256 = (
    "3348fdfb989fb85ad3598aae2b0bd3d9faf0b62aebdfa30aa6b4b1dc16d8b6a7"
)
MASKING = {
    "type": "partial_gene_expression",
    "rate": 0.5,
    "rates": {"common_evaluation": 0.2, "native_evaluation": 0.5},
    "mask_expression_only": True,
}
PILOT_THRESHOLDS = {
    "peak_allocated_vram_gib_maximum": 20.5,
    "peak_host_memory_gib_maximum": 40.0,
    "projected_200_epoch_runtime_hours_maximum": 6.0,
    "projected_final_free_disk_gib_minimum": 27.5,
}


class MyJJuMaterializationError(RuntimeError):
    """Raised when preparation would violate the frozen campaign."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MyJJuMaterializationError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise MyJJuMaterializationError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MyJJuMaterializationError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject,
            object_pairs_hook=unique,
        )
    except MyJJuMaterializationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MyJJuMaterializationError(f"{label} is unreadable") from exc
    return dict(_mapping(value, label))


def _signed_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    payload = _strict_json(path, label=label)
    checksum = payload.get("checksum")
    core = dict(payload)
    core.pop("checksum", None)
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or canonical_sha256(core) != checksum
    ):
        raise MyJJuMaterializationError(f"{label} checksum does not verify")
    return payload, checksum


def _component(reference: str, section: str) -> tuple[dict[str, Any], str]:
    path = (PROJECT_ROOT / reference).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise MyJJuMaterializationError(
            f"{section} component escapes the project root"
        ) from exc
    if not path.is_file() or path.is_symlink():
        raise MyJJuMaterializationError(
            f"{section} component is missing or unsafe: {reference}"
        )
    payload = load_yaml_mapping(path)
    if set(payload) != {section}:
        raise MyJJuMaterializationError(
            f"{reference} must contain only the {section!r} group"
        )
    return deepcopy(dict(_mapping(payload[section], reference))), sha256_file(path)


def _project_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise MyJJuMaterializationError(
            f"path must remain under project root: {path}"
        ) from exc


def _verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = PROJECT_ROOT / CONTRACT_RELATIVE
    campaign_path = PROJECT_ROOT / CAMPAIGN_RELATIVE
    source_audit_path = PROJECT_ROOT / SOURCE_AUDIT_RELATIVE
    if (
        sha256_file(contract_path) != CONTRACT_SHA256
        or sha256_file(source_audit_path) != SOURCE_AUDIT_SHA256
    ):
        raise MyJJuMaterializationError(
            "frozen contract or external-source audit checksum changed"
        )
    contract = load_yaml_mapping(contract_path)
    campaign = load_yaml_mapping(campaign_path)
    source_audit = load_yaml_mapping(source_audit_path)
    if (
        contract.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("frozen_contract_sha256") != CONTRACT_SHA256
        or source_audit.get("audited_before_ten_core_execution") is not True
    ):
        raise MyJJuMaterializationError(
            "campaign declaration disagrees with its frozen inputs"
        )
    cohort = _mapping(contract.get("cohort"), "contract.cohort")
    training = _mapping(contract.get("training"), "contract.training")
    model = _mapping(contract.get("model"), "contract.model")
    if (
        tuple(cohort.get("aliases", ())) != ANC_ALIASES
        or cohort.get("total_fit_cells") != EXPECTED_TOTAL_NODES
        or cohort.get("genes") != EXPECTED_N_GENES
        or tuple(cohort.get("model_seeds", ())) != MODEL_SEEDS
        or model.get("expected_trainable_parameters")
        != EXPECTED_PARAMETER_COUNT
        or training.get("global_epochs") != 200
        or training.get("train_mask_rate") != 0.5
    ):
        raise MyJJuMaterializationError("frozen scientific contract drifted")
    return dict(contract), dict(source_audit)


def _verify_external_source(source_audit: Mapping[str, Any]) -> dict[str, Any]:
    repository = _mapping(source_audit.get("repository"), "source repository")
    checkout = Path(str(repository.get("checkout", ""))).resolve()
    if not checkout.is_dir() or checkout.is_symlink():
        raise MyJJuMaterializationError(
            "audited external source checkout is missing or unsafe"
        )
    try:
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MyJJuMaterializationError(
            "external source Git identity cannot be verified"
        ) from exc
    if commit != repository.get("commit"):
        raise MyJJuMaterializationError("external source commit changed")
    source_files = _mapping(source_audit.get("source_files"), "source files")
    verified: dict[str, Any] = {}
    for label, raw in source_files.items():
        record = _mapping(raw, f"source file {label}")
        relative = Path(str(record.get("relative_path", "")))
        path = (checkout / relative).resolve()
        try:
            path.relative_to(checkout)
        except ValueError as exc:
            raise MyJJuMaterializationError(
                f"source file {label} escapes the checkout"
            ) from exc
        checksum = sha256_file(path)
        if checksum != record.get("sha256"):
            raise MyJJuMaterializationError(
                f"external source checksum changed for {label}"
            )
        verified[str(label)] = {
            "relative_path": relative.as_posix(),
            "sha256": checksum,
        }
    return {
        "checkout": str(checkout),
        "commit": commit,
        "branch": repository.get("branch"),
        "files": verified,
        "source_audit_reference": SOURCE_AUDIT_RELATIVE.as_posix(),
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
    }


def _prepared_paths(dataset: Mapping[str, Any]) -> tuple[tuple[str, Path], ...]:
    raw = _mapping(dataset.get("prepared_artifacts"), "prepared artifacts")
    if tuple(raw) != ANC_ALIASES:
        raise MyJJuMaterializationError(
            "prepared artifact aliases are not exact and ordered"
        )
    result: list[tuple[str, Path]] = []
    for alias in ANC_ALIASES:
        path = (PROJECT_ROOT / str(raw[alias])).resolve()
        try:
            path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise MyJJuMaterializationError(
                f"{alias} prepared artifact escapes project root"
            ) from exc
        if not path.is_dir() or path.is_symlink():
            raise MyJJuMaterializationError(
                f"{alias} prepared artifact is missing or unsafe"
            )
        result.append((alias, path))
    return tuple(result)


def _cohort_identity(
    dataset: Mapping[str, Any],
    *,
    registered_identity: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    cohort = load_pooled_full_core_cohort(_prepared_paths(dataset))
    if (
        tuple(cohort.aliases) != ANC_ALIASES
        or cohort.total_nodes != EXPECTED_TOTAL_NODES
        or cohort.n_genes != EXPECTED_N_GENES
        or any(
            core.preprocessing_qc.protected_identifier_arrays_returned
            for core in cohort.cores
        )
    ):
        raise MyJJuMaterializationError("verified ten-core cohort drifted")
    split_basis = dict(
        _mapping(
            registered_identity.get("split_fingerprint_basis"),
            "registered split fingerprint basis",
        )
    )
    if (
        registered_identity.get("dataset_fingerprint")
        != cohort.fingerprint_sha256
        or registered_identity.get("split_id")
        != "1331b7ed9f3b2d73"
        or registered_identity.get("split_fingerprint")
        != "1331b7ed9f3b2d7370b9ae3ada94b3af11364fddccbf58cb8979b7adf71e936a"
        or split_basis.get("dataset_fingerprint")
        != cohort.fingerprint_sha256
        or split_basis.get("role_counts")
        != {"fit": EXPECTED_TOTAL_NODES, "validation": 0, "test": 0}
        or canonical_sha256(split_basis)
        != registered_identity.get("split_fingerprint")
    ):
        raise MyJJuMaterializationError(
            "authoritative pooled dataset/split identity changed"
        )
    identity = {
        "aliases": list(cohort.aliases),
        "total_nodes": cohort.total_nodes,
        "n_genes": cohort.n_genes,
        "ordered_gene_schema_sha256": (
            cohort.checksums.ordered_gene_schema_sha256
        ),
        "dataset_fingerprint": cohort.fingerprint_sha256,
        "cohort_checksums": cohort.checksums.to_dict(),
        "split_id": str(registered_identity["split_id"]),
        "split_fingerprint": str(
            registered_identity["split_fingerprint"]
        ),
        "split_fingerprint_basis": split_basis,
        "cores": {
            core.alias: {
                "n_nodes": core.n_nodes,
                "n_genes": core.n_genes,
                "checksums": core.checksums.to_dict(),
            }
            for core in cohort.cores
        },
    }
    return cohort, identity


def _comparator_masks() -> tuple[dict[str, Any], str, dict[str, Any]]:
    path = PROJECT_ROOT / COMPARATOR_RECEIPT_RELATIVE
    if sha256_file(path) != COMPARATOR_RECEIPT_FILE_SHA256:
        raise MyJJuMaterializationError(
            "pinned current BAGM materialization file checksum changed"
        )
    payload, checksum = _signed_json(
        path, label="current BAGM materialization receipt"
    )
    if (
        checksum != COMPARATOR_RECEIPT_CHECKSUM
        or
        payload.get("campaign_id") != COMPARATOR_CAMPAIGN_ID
        or tuple(
            _mapping(payload.get("evaluation_mask_sources"), "mask sources")
        )
        != ANC_ALIASES
    ):
        raise MyJJuMaterializationError(
            "current BAGM mask materialization is incompatible"
        )
    result: dict[str, Any] = {}
    for alias in ANC_ALIASES:
        record = dict(
            _mapping(
                payload["evaluation_mask_sources"][alias],
                f"{alias} BAGM mask source",
            )
        )
        partial = [
            item
            for item in record.get("entries", ())
            if item.get("mode") == "partial_gene"
        ]
        if (
            len(partial) != 3
            or {item.get("replicate") for item in partial} != {0, 1, 2}
            or any(len(str(item.get("mask_checksum", ""))) != 64 for item in partial)
        ):
            raise MyJJuMaterializationError(
                f"{alias} BAGM partial-gene mask identities are incomplete"
            )
        result[alias] = record
    return result, checksum, dict(
        _mapping(payload.get("cohort"), "registered comparator cohort")
    )


def _graph_and_mask_identity(
    cohort: Any,
    comparator_masks: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Imported lazily so unit tests can exercise pure materialization helpers
    # without requiring torch-geometric at collection time.
    from scripts.train.run_myjju_genemae_pooled import (
        prepare_source_tiles,
        regenerate_evaluation_masks,
    )

    tiled = prepare_source_tiles(cohort, k=15, max_nodes=7000)
    graphs = tiled.identity()
    common: dict[str, Any] = {}
    for core in cohort.cores:
        masks = regenerate_evaluation_masks(
            core,
            comparator_source=comparator_masks[core.alias],
            native_base_seed=None,
        )
        common[core.alias] = masks.identity()
    return graphs, common


def _parameter_audit() -> dict[str, Any]:
    from spatial_benchmark.myjju_genemae import count_parameters, make_source_model

    model = make_source_model(num_genes=EXPECTED_N_GENES, mask_rate=0.5)
    count = count_parameters(model)
    shapes = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if count != EXPECTED_PARAMETER_COUNT:
        raise MyJJuMaterializationError(
            f"source-fidelity parameter count changed: {count}"
        )
    return {
        "trainable_parameter_count": count,
        "expected_trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
        "exact_match": True,
        "named_parameter_shapes_sha256": canonical_sha256(shapes),
    }


def _dataset_section(
    component: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = deepcopy(dict(component))
    dataset.update(
        {
            "task": "masked_expression_regression",
            "target_scale": "full_cell_log1p_cp10k",
            # This is the registered input-materialization version.  CP10k is
            # a model-side transform and is recorded below and in features.
            "preprocessing_version": component["preprocessing_version"],
            "split_id": identity["split_id"],
            "dataset_fingerprint": identity["dataset_fingerprint"],
            "dataset_fingerprint_role": "verified_source_content",
            "split_fingerprint": identity["split_fingerprint"],
            "split_fingerprint_status": (
                "verified_materialized_no_holdout_role_assignment"
            ),
            "split_fingerprint_basis": deepcopy(
                identity["split_fingerprint_basis"]
            ),
            "ordered_gene_schema_sha256": (
                identity["ordered_gene_schema_sha256"]
            ),
            "core_node_counts": {
                alias: identity["cores"][alias]["n_nodes"]
                for alias in ANC_ALIASES
            },
            "count_representation": {
                "source_scale": "raw_biological_probe_counts",
                "transform": "full_cell_log1p_cp10k_before_masking",
                "target_sum": 10000.0,
                "hidden_entries_contribute_to_library_denominator": True,
            },
        }
    )
    return dataset


def _build_config(
    *,
    role: str,
    seed: int,
    requested_gpu: int,
    output_reference: str,
    components: Mapping[str, Mapping[str, Any]],
    dataset: Mapping[str, Any],
    cohort_identity: Mapping[str, Any],
    graph_identity: Mapping[str, Any],
    mask_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    pilot = role == "pilot"
    graph = deepcopy(dict(components["graph"]))
    graph["expected_tiled_graphs"] = deepcopy(dict(graph_identity))
    evaluation = deepcopy(dict(components[f"evaluation_{role}"]))
    evaluation["prior_mask_sources"] = deepcopy(dict(mask_identity))
    evaluation["native_mask_seed_namespace"] = (
        "myjju-genemae-native-50-v1"
    )
    launcher = deepcopy(dict(components["launcher"]))
    launcher["requested_gpu"] = str(requested_gpu)
    config = {
        "model": deepcopy(dict(components["model"])),
        "masking": deepcopy(MASKING),
        "dataset": _dataset_section(dataset, cohort_identity),
        "features": deepcopy(dict(components["features"])),
        "graph": graph,
        "trainer": deepcopy(dict(components[f"trainer_{role}"])),
        "evaluation": evaluation,
        "launcher": launcher,
        "version": 1,
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "display_name": "MyJJu GeneMAE ten-core comparison",
            "exploratory": True,
            "frozen_contract": CONTRACT_RELATIVE.as_posix(),
            "frozen_contract_sha256": CONTRACT_SHA256,
        },
        "experiment": {
            "variant_label": (
                "myjju_genemae_resource_pilot"
                if pilot
                else "myjju_genemae_ensemble_member"
            ),
            "arm": "myjju-genemae",
            "core_aliases": list(ANC_ALIASES),
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "estimand": (
                "implementation_resource_feasibility"
                if pilot
                else "held_in_partial_gene_reconstruction"
            ),
            "permitted_claim": (
                "diagnostic_runtime_memory_and_finiteness_only"
                if pilot
                else "descriptive_held_in_partial_gene_architecture_comparison"
            ),
            "resource_pilot": pilot,
            "conclusion_eligible": not pilot,
            "cross_core_edges": False,
            "cross_tile_edges": False,
        },
        "classification": {
            "schema_version": 1,
            "lifecycle_stage": "diagnostic" if pilot else "exploratory_screen",
            "study_axis": "external_architecture_reproduction",
            "scientific_variant": "myjju-genemae-source-fidelity",
            "retention_class": (
                "retain_diagnostic_evidence"
                if pilot
                else "retain_exploratory_evidence"
            ),
            "classification_confidence": "high",
        },
        "metadata": {
            "locked_config_materialization_receipt": (
                f"{output_reference}/{RECEIPT_NAME}"
            ),
            "frozen_scientific_contract": True,
            "execution_role": "resource_pilot" if pilot else "production",
            "source_audit_reference": SOURCE_AUDIT_RELATIVE.as_posix(),
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "external_source_commit": source_identity["commit"],
            "production_requires_pilot_gate": not pilot,
            "pilot_gate_receipt_reference": (
                f"{output_reference}/{PILOT_GATE_NAME}"
            ),
        },
        "seed": seed,
        "fold": 0,
        "attempt": 1,
    }
    validate_experiment_config(config)
    command = command_for_config(config)
    if not any(part.endswith("run_myjju_genemae_pooled.py") for part in command):
        raise MyJJuMaterializationError(
            "resolved config does not derive the MyJJu queue-owned runner"
        )
    if requested_gpu not in SAFE_GPU_IDS:
        raise MyJJuMaterializationError("resolved config requests an unsafe GPU")
    return config


def _yaml_bytes(value: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(value), sort_keys=True, allow_unicode=False
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_table_rows(root: Path, relative_stem: str) -> list[dict[str, Any]]:
    candidates = [
        root / f"{relative_stem}{suffix}"
        for suffix in (".parquet", ".jsonl", ".csv")
        if (root / f"{relative_stem}{suffix}").is_file()
    ]
    if len(candidates) != 1:
        raise MyJJuMaterializationError(
            f"expected one table representation for {relative_stem}"
        )
    path = candidates[0]
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise MyJJuMaterializationError(
                "pyarrow is required to verify pilot Parquet coverage"
            ) from exc
        return parquet.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [
            dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _existing_matches(output: Path, files: Mapping[str, bytes]) -> bool:
    if not output.is_dir() or output.is_symlink():
        return False
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and path.relative_to(output).as_posix() != PILOT_GATE_NAME
    }
    return observed == set(files) and all(
        (output / relative).read_bytes() == content
        for relative, content in files.items()
    )


def _publish_atomically(output: Path, files: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent)
    )
    try:
        for relative, content in files.items():
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if not _existing_matches(stage, files):
            raise MyJJuMaterializationError(
                "staged materialization failed byte verification"
            )
        if output.exists() or output.is_symlink():
            if _existing_matches(output, files):
                return
            raise MyJJuMaterializationError(
                "locked output exists with different content"
            )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def materialize_campaign(
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish one pilot and seven production configs."""

    _contract, source_audit = _verify_frozen_inputs()
    source_identity = _verify_external_source(source_audit)
    sections: dict[str, Mapping[str, Any]] = {}
    hashes: dict[str, str] = {}
    for section, reference in COMPONENTS.items():
        sections[section], hashes[reference] = _component(reference, section)
    dataset, hashes[DATASET_COMPONENT] = _component(
        DATASET_COMPONENT, "dataset"
    )
    for role in ("pilot", "production"):
        reference = TRAINERS[role]
        sections[f"trainer_{role}"], hashes[reference] = _component(
            reference, "trainer"
        )
        reference = EVALUATIONS[role]
        sections[f"evaluation_{role}"], hashes[reference] = _component(
            reference, "evaluation"
        )
    parameter_audit = _parameter_audit()
    implementation_hashes = {
        label: {
            "reference": reference,
            "sha256": sha256_file(PROJECT_ROOT / reference),
        }
        for label, reference in IMPLEMENTATION_FILES.items()
    }
    if (
        implementation_hashes["model_module"]["sha256"]
        != PINNED_MODEL_MODULE_SHA256
    ):
        raise MyJJuMaterializationError(
            "pinned MyJJu compatibility module checksum changed"
        )
    (
        comparator_masks,
        comparator_checksum,
        registered_identity,
    ) = _comparator_masks()
    cohort, cohort_identity = _cohort_identity(
        dataset,
        registered_identity=registered_identity,
    )
    graph_identity, mask_identity = _graph_and_mask_identity(
        cohort, comparator_masks
    )
    canonical_destination = (PROJECT_ROOT / LOCKED_RELATIVE).resolve()
    destination = (
        output_dir.resolve()
        if output_dir is not None
        else canonical_destination
    )
    if destination != canonical_destination:
        raise MyJJuMaterializationError(
            "custom output directories are incompatible with the queue-owned "
            "runner's canonical locked receipt reference"
        )
    output_reference = _project_reference(destination)
    generated: dict[str, bytes] = {}
    jobs: list[dict[str, Any]] = []
    plans = [
        ("pilot", 0, PILOT_GPU_ID),
        *[("production", seed, SEED_GPU_MAP[seed]) for seed in MODEL_SEEDS],
    ]
    for role, seed, requested_gpu in plans:
        config = _build_config(
            role=role,
            seed=seed,
            requested_gpu=requested_gpu,
            output_reference=output_reference,
            components=sections,
            dataset=dataset,
            cohort_identity=cohort_identity,
            graph_identity=graph_identity,
            mask_identity=mask_identity,
            source_identity=source_identity,
        )
        relative = (
            f"{role}_configs/"
            f"seed-{seed:02d}_myjju_genemae"
            f"{'_resource_pilot' if role == 'pilot' else ''}.yaml"
        )
        content = _yaml_bytes(config)
        generated[relative] = content
        jobs.append(
            {
                "role": role,
                "seed": seed,
                "requested_gpu": requested_gpu,
                "config": f"{output_reference}/{relative}",
                "config_sha256": canonical_sha256(config),
                "file_sha256": _sha256_bytes(content),
                "enqueue_authorized": role == "pilot",
                "blocked_on_pilot_gate": role == "production",
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": RECEIPT_KIND,
        "campaign_id": CAMPAIGN_ID,
        "frozen_contract": {
            "reference": CONTRACT_RELATIVE.as_posix(),
            "sha256": CONTRACT_SHA256,
        },
        "external_source": source_identity,
        "component_file_sha256": dict(sorted(hashes.items())),
        "implementation_file_sha256": implementation_hashes,
        "parameter_audit": parameter_audit,
        "cohort": cohort_identity,
        "tiled_graphs": graph_identity,
        "evaluation_masks": mask_identity,
        "comparator_materialization": {
            "campaign_id": COMPARATOR_CAMPAIGN_ID,
            "reference": COMPARATOR_RECEIPT_RELATIVE.as_posix(),
            "canonical_checksum": comparator_checksum,
            "file_sha256": COMPARATOR_RECEIPT_FILE_SHA256,
        },
        "assignment": {
            "allowed_gpu_ids": list(SAFE_GPU_IDS),
            "excluded_gpu_ids": [4],
            "pilot_gpu_id": PILOT_GPU_ID,
            "seed_gpu_map": {
                str(seed): SEED_GPU_MAP[seed] for seed in MODEL_SEEDS
            },
        },
        "pilot_thresholds": deepcopy(PILOT_THRESHOLDS),
        "pilot_gate_receipt_reference": f"{output_reference}/{PILOT_GATE_NAME}",
        "jobs": jobs,
        "counts": {
            "pilot_configs": 1,
            "production_configs": 7,
            "production_seeds": 7,
            "cores": 10,
        },
        "queue_mutation_performed": False,
        "registry_mutation_performed": False,
        "training_performed": False,
    }
    receipt["checksum"] = canonical_sha256(receipt)
    generated[RECEIPT_NAME] = _json_bytes(receipt)
    enqueue_plan = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "materialization_checksum": receipt["checksum"],
        "pilot": [job for job in jobs if job["role"] == "pilot"],
        "production": [job for job in jobs if job["role"] == "production"],
        "production_blocked_until": PILOT_GATE_NAME,
        "queue_mutation_performed": False,
        "registry_mutation_performed": False,
    }
    generated["enqueue_plan.json"] = _json_bytes(enqueue_plan)
    if destination.exists() or destination.is_symlink():
        if not _existing_matches(destination, generated):
            raise MyJJuMaterializationError(
                "locked materialization already exists with different content"
            )
        return receipt
    _publish_atomically(destination, generated)
    if not _existing_matches(destination, generated):
        raise MyJJuMaterializationError(
            "published materialization failed verification"
        )
    return receipt


def materialize_pilot_gate_receipt(
    *,
    pilot_run: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify a successful pilot bundle and exclusively publish its gate."""

    root = pilot_run.resolve()
    verification = verify_run_bundle(root, require_success_contract=True)
    summary = _strict_json(root / "summary.json", label="pilot summary")
    resource = _strict_json(
        root / "diagnostics/resource.json", label="pilot resource diagnostic"
    )
    convergence = _strict_json(
        root / "diagnostics/convergence.json",
        label="pilot convergence diagnostic",
    )
    final_metrics = _strict_json(
        root / "metrics/final.json", label="pilot final metrics"
    )
    hardware = _strict_json(
        root / "provenance/hardware.json", label="pilot hardware provenance"
    )
    resolved_config = dict(
        load_yaml_mapping(root / "config.resolved.yaml")
    )
    validate_experiment_config(resolved_config)
    if (
        verification.get("status") != "success"
        or summary.get("campaign_id") != CAMPAIGN_ID
        or summary.get("status") != "success"
        or summary.get("model_name") != "myjju-genemae"
        or summary.get("diagnostic_resource_pilot") is not True
        or summary.get("model_seed") != 0
        or summary.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or summary.get("final_epoch") != 1
        or summary.get("completed_global_epochs") != 2
        or not isinstance(summary.get("materialization_checksum"), str)
        or str(hardware.get("requested_gpu")) != str(PILOT_GPU_ID)
        or not str(resource.get("device", "")).startswith("cuda")
        or resolved_config.get("seed") != 0
        or resolved_config.get("campaign", {}).get("campaign_id")
        != CAMPAIGN_ID
        or resolved_config.get("model", {}).get("name")
        != "myjju-genemae"
        or resolved_config.get("metadata", {}).get("execution_role")
        != "resource_pilot"
        or resolved_config.get("trainer", {}).get("max_epochs") != 2
        or resolved_config.get("evaluation", {}).get("primary_metric")
        != "fit/partial_gene/log1p_cp10k_masked_huber"
    ):
        raise MyJJuMaterializationError(
            "pilot bundle does not satisfy the frozen resource gate"
        )
    canonical_destination = (PROJECT_ROOT / LOCKED_RELATIVE).resolve()
    destination = (
        output_dir.resolve()
        if output_dir is not None
        else canonical_destination
    )
    if destination != canonical_destination:
        raise MyJJuMaterializationError(
            "pilot gates must use the canonical locked campaign directory"
        )
    materialization, checksum = _signed_json(
        destination / RECEIPT_NAME, label="MyJJu materialization receipt"
    )
    if materialization.get("campaign_id") != CAMPAIGN_ID:
        raise MyJJuMaterializationError("pilot gate materialization mismatch")
    if summary.get("materialization_checksum") != checksum:
        raise MyJJuMaterializationError(
            "pilot summary is bound to a different materialization"
        )
    implementation = _mapping(
        materialization.get("implementation_file_sha256"),
        "materialized implementation checksums",
    )
    for label, reference in IMPLEMENTATION_FILES.items():
        record = _mapping(
            implementation.get(label), f"implementation {label}"
        )
        if (
            record.get("reference") != reference
            or record.get("sha256")
            != sha256_file(PROJECT_ROOT / reference)
        ):
            raise MyJJuMaterializationError(
                f"pilot implementation checksum changed for {label}"
            )
    pilot_jobs = [
        record
        for record in materialization.get("jobs", ())
        if record.get("role") == "pilot" and record.get("seed") == 0
    ]
    if len(pilot_jobs) != 1:
        raise MyJJuMaterializationError(
            "materialized pilot job slot is not unique"
        )
    pilot_job = dict(pilot_jobs[0])
    source_config_path = PROJECT_ROOT / str(pilot_job.get("config", ""))
    if (
        not source_config_path.is_file()
        or sha256_file(source_config_path) != pilot_job.get("file_sha256")
    ):
        raise MyJJuMaterializationError(
            "materialized pilot config checksum changed"
        )
    source_config = dict(load_yaml_mapping(source_config_path))
    normalized_runtime = dict(resolved_config)
    normalized_source = dict(source_config)
    normalized_runtime["attempt"] = 1
    normalized_source["attempt"] = 1
    if canonical_sha256(normalized_runtime) != canonical_sha256(
        normalized_source
    ):
        raise MyJJuMaterializationError(
            "resolved pilot config differs from its materialized job slot"
        )
    required_metric = "fit/partial_gene/log1p_cp10k_masked_huber"
    primary_value = final_metrics.get(required_metric)
    numeric_resource_fields = (
        "peak_allocated_vram_gib",
        "peak_host_memory_gib",
        "projected_200_epoch_runtime_hours",
        "projected_final_free_disk_gib",
    )
    if (
        primary_value is None
        or not math.isfinite(float(primary_value))
        or any(
            resource.get(field) is None
            or not math.isfinite(float(resource[field]))
            for field in numeric_resource_fields
        )
    ):
        raise MyJJuMaterializationError(
            "pilot primary metric or resource facts are non-finite"
        )
    metric_rows = _read_table_rows(root, "metrics/per_core_replicate")
    from scripts.train.run_myjju_genemae_pooled import (
        REQUIRED_EVALUATION_METRIC_FIELDS,
        evaluation_metric_audit_rows,
        reconstruct_model_from_checkpoint,
        state_dict_sha256,
    )

    try:
        metric_audit_rows = evaluation_metric_audit_rows(metric_rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise MyJJuMaterializationError(
            "pilot per-core replicate metric table is malformed"
        ) from exc
    evaluation_coverage = _strict_json(
        root / "diagnostics/evaluation_coverage.json",
        label="pilot evaluation coverage",
    )
    metrics_finite = all(
        math.isfinite(float(row[field]))
        for row in metric_audit_rows
        for field in REQUIRED_EVALUATION_METRIC_FIELDS
    )
    metric_audit_sha256 = canonical_sha256(metric_audit_rows)
    observed_slots = {
        (
            str(row["biological_unit_alias"]),
            str(row["mask_label"]),
            int(row["mask_replicate"]),
            str(row["graph_condition"]),
        )
        for row in metric_audit_rows
    }
    expected_slots = {
        (alias, label, replicate, condition)
        for alias in ANC_ALIASES
        for label in ("common_20", "native_50")
        for replicate in range(3)
        for condition in ("observed", "node_label_permuted")
    }
    recomputed_runtime_hours = (
        float(resource["data_duration_seconds"])
        + float(resource["graph_duration_seconds"])
        + 200.0 * float(resource["mean_epoch_duration_seconds"])
        + float(resource["evaluation_duration_seconds"])
        + float(
            resource[
                "measured_checkpoint_and_replay_finalization_allowance_seconds"
            ]
        )
    ) / 3600.0
    recomputed_final_free_disk_gib = max(
        0.0,
        float(resource["current_free_disk_gib"])
        - float(resource["projected_campaign_output_gib"]),
    )
    independent_checks = {
        "peak_vram_passed": float(resource["peak_allocated_vram_gib"])
        <= PILOT_THRESHOLDS["peak_allocated_vram_gib_maximum"],
        "peak_host_memory_passed": float(resource["peak_host_memory_gib"])
        <= PILOT_THRESHOLDS["peak_host_memory_gib_maximum"],
        "projected_runtime_passed": float(
            resource["projected_200_epoch_runtime_hours"]
        )
        <= PILOT_THRESHOLDS[
            "projected_200_epoch_runtime_hours_maximum"
        ],
        "projected_disk_passed": float(
            resource["projected_final_free_disk_gib"]
        )
        >= PILOT_THRESHOLDS["projected_final_free_disk_gib_minimum"],
        "projected_runtime_arithmetic_reconciles": math.isclose(
            float(resource["projected_200_epoch_runtime_hours"]),
            recomputed_runtime_hours,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "projected_disk_arithmetic_reconciles": math.isclose(
            float(resource["projected_final_free_disk_gib"]),
            recomputed_final_free_disk_gib,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "all_global_epochs_completed": convergence.get(
            "all_global_epochs_completed"
        )
        is True,
        "all_losses_and_gradients_finite": convergence.get(
            "all_losses_and_gradients_finite"
        )
        is True,
        "all_parameters_finite": convergence.get("all_parameters_finite")
        is True,
        "checkpoint_replay_passed": _strict_json(
            root / "diagnostics/checkpoint_replay.json",
            label="pilot checkpoint replay",
        ).get("replay_passed")
        is True,
        "full_evaluation_coverage": (
            summary.get("evaluation_row_count") == 120
            and len(metric_rows) == 120
            and len(metric_audit_rows) == 120
            and observed_slots == expected_slots
            and metrics_finite
            and evaluation_coverage.get("row_count") == 120
            and evaluation_coverage.get("all_required_metrics_finite")
            is True
            and evaluation_coverage.get("metric_audit_sha256")
            == metric_audit_sha256
        ),
    }
    if not all(independent_checks.values()):
        failed = sorted(
            name for name, passed in independent_checks.items() if not passed
        )
        raise MyJJuMaterializationError(
            "pilot independently fails frozen gates: " + ", ".join(failed)
        )
    checkpoint_path = root / "checkpoints/last.ckpt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    replay_model = reconstruct_model_from_checkpoint(
        checkpoint, device="cpu"
    )
    if (
        checkpoint.get("final_epoch") != 1
        or checkpoint.get("seed") != 0
        or checkpoint.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or checkpoint.get("run_id") != root.name
        or checkpoint.get("materialization_checksum") != checksum
        or state_dict_sha256(replay_model.state_dict())
        != checkpoint.get("state_dict_sha256")
        or checkpoint.get("implementation_file_sha256")
        != implementation
    ):
        raise MyJJuMaterializationError(
            "pilot final checkpoint identity or strict reload changed"
        )
    del replay_model
    marker_sha = sha256_file(root / "_SUCCESS")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": PILOT_GATE_KIND,
        "campaign_id": CAMPAIGN_ID,
        "materialization_checksum": checksum,
        "pilot_run_id": root.name,
        "pilot_artifact_reference": _project_reference(root),
        "success_marker_sha256": marker_sha,
        "resource_diagnostic_sha256": sha256_file(
            root / "diagnostics/resource.json"
        ),
        "checkpoint_sha256": sha256_file(root / "checkpoints/last.ckpt"),
        "resolved_pilot_config_sha256": canonical_sha256(resolved_config),
        "materialized_pilot_config_reference": str(pilot_job["config"]),
        "materialized_pilot_config_file_sha256": str(
            pilot_job["file_sha256"]
        ),
        "materialized_pilot_job_slot": {
            "role": "pilot",
            "seed": 0,
            "requested_gpu": pilot_job["requested_gpu"],
        },
        "thresholds": deepcopy(PILOT_THRESHOLDS),
        "observed": {
            key: resource[key]
            for key in (
                "peak_allocated_vram_gib",
                "peak_host_memory_gib",
                "projected_200_epoch_runtime_hours",
                "projected_final_free_disk_gib",
            )
        },
        "independent_checks": independent_checks,
        "runner_declared_pilot_gate_passed": resource.get(
            "pilot_gate_passed"
        ),
        "gate_passed": True,
        "queue_mutation_performed": False,
        "registry_mutation_performed": False,
    }
    receipt["checksum"] = canonical_sha256(receipt)
    path = destination / PILOT_GATE_NAME
    content = _json_bytes(receipt)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == content:
            return receipt
        raise MyJJuMaterializationError(
            "pilot gate receipt exists with different content"
        )
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise MyJJuMaterializationError(
            "pilot gate receipt appeared during exclusive publication"
        ) from exc
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action")
    materialize = subparsers.add_parser(
        "materialize", help="materialize configs without enqueuing"
    )
    verify = subparsers.add_parser(
        "verify-pilot", help="materialize a production gate from a pilot bundle"
    )
    verify.add_argument("--pilot-run", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    action = args.action or "materialize"
    if action == "verify-pilot":
        receipt = materialize_pilot_gate_receipt(
            pilot_run=args.pilot_run,
            output_dir=None,
        )
        output = {
            "campaign_id": CAMPAIGN_ID,
            "action": action,
            "pilot_run_id": receipt["pilot_run_id"],
            "checksum": receipt["checksum"],
            "gate_passed": True,
        }
    else:
        receipt = materialize_campaign(output_dir=None)
        output = {
            "campaign_id": CAMPAIGN_ID,
            "action": action,
            "counts": receipt["counts"],
            "checksum": receipt["checksum"],
            "queue_mutation_performed": False,
            "registry_mutation_performed": False,
            "training_performed": False,
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

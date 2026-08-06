#!/usr/bin/env python3
"""Materialize the locked pooled ten-core hybrid-count campaign.

This entry point is preparation-only.  It validates the frozen contract, the
ten alias-safe prepared artifacts, the prior verified graph and fixed-mask
identities, and exact GAT/self parameter matching.  It atomically publishes
two resource-pilot and fourteen production resolved configurations plus a
checksum-bound receipt.  It never registers, enqueues, or trains a run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS  # noqa: E402
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.full_core import EDGE_ATTRIBUTE_NAMES  # noqa: E402
from spatial_benchmark.hybrid_count import (  # noqa: E402
    HybridEdgeParameterMatchedSelfControl,
    HybridReceiverChunkedEdgeConditionedGATv2,
    assert_exact_parameter_match,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.pooled_full_core import (  # noqa: E402
    ANC_ALIASES,
    EXPECTED_N_GENES,
    EXPECTED_PREPARED_CORES,
    EXPECTED_TOTAL_NODES,
    load_pooled_full_core_cohort,
)
from spatial_benchmark.queueing import command_for_config  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


CAMPAIGN_ID = "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
PRIOR_CAMPAIGN_ID = "cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
ALIASES = tuple(ANC_ALIASES)
ARMS = ("pooled-hybrid-gat-k1000", "pooled-hybrid-matched-self")
MODEL_SEEDS = tuple(range(7))
SAFE_GPU_IDS = (0, 1, 2, 3, 5, 6, 7)
SEED_GPU_MAP = dict(zip(MODEL_SEEDS, SAFE_GPU_IDS, strict=True))
PILOT_GPU_MAP = {
    "pooled-hybrid-gat-k1000": 0,
    "pooled-hybrid-matched-self": 1,
}
EXPECTED_PARAMETER_COUNT = 11_674_880
EXPECTED_FROZEN_CONTRACT_SHA256 = (
    "c6af3dc756155ee502506f08304a7436ae99da36ad2b4ed8fae48672a312f6e2"
)
EXPECTED_PRIOR_MATERIALIZATION_CHECKSUM = (
    "ff859d8d5358a6f855d38f428e225537022af7d746bde4d277012292087d8b35"
)
RECEIPT_NAME = "locked_config_materialization.json"
RECEIPT_KIND = "pooled_hybrid_count_locked_config_materialization_v1"
PILOT_GATE_RECEIPT_NAME = "pilot_gate_receipt.json"
PILOT_GATE_RECEIPT_KIND = "pooled_hybrid_count_pilot_gate_v1"

_PRIOR_GAT_RUN_IDS = {
    "ANC-01": "r_20260729T070251Z_3c2e7600_s000_f00_a01_b09a7537",
    "ANC-02": "r_20260729T070254Z_f2addf2d_s000_f00_a01_f87da652",
    "ANC-03": "r_20260729T070252Z_c7825650_s000_f00_a01_94a5c22d",
    "ANC-04": "r_20260729T070254Z_8f494e63_s000_f00_a01_40829054",
    "ANC-05": "r_20260729T071943Z_a49ef794_s000_f00_a01_bd87d846",
    "ANC-06": "r_20260729T070254Z_06e4a0ef_s000_f00_a01_e72a8ec4",
    "ANC-07": "r_20260729T070459Z_5f3bbb10_s000_f00_a01_a235dd64",
    "ANC-08": "r_20260729T071150Z_aa18447c_s000_f00_a01_3b0008a8",
    "ANC-09": "r_20260729T070459Z_f2296cb4_s000_f00_a01_25c538fa",
    "ANC-10": "r_20260729T071357Z_d0b3fadc_s000_f00_a01_f81f5f4a",
}

_COMPONENTS = {
    "dataset": "configs/dataset/adjacent_normal_10core_pooled_fit_v1.yaml",
    "features": "configs/features/cosmx_full_core_morphology_edge_geometry.yaml",
    "graph": "configs/graph/pooled_k1000_r2000_mutual.yaml",
    "launcher": "configs/launcher/local_single_gpu_3090.yaml",
}
_MODEL_COMPONENTS = {
    "pooled-hybrid-gat-k1000": "configs/model/hybrid_count_gat.yaml",
    "pooled-hybrid-matched-self": (
        "configs/model/hybrid_count_matched_self.yaml"
    ),
}
_TRAINER_COMPONENTS = {
    "pilot": "configs/trainer/pooled_hybrid_resource_pilot_2.yaml",
    "production": "configs/trainer/pooled_hybrid_fixed_200.yaml",
}
_EVALUATION_COMPONENTS = {
    "pilot": (
        "configs/evaluation/"
        "held_in_pooled_10core_hybrid_count_pilot_v1.yaml"
    ),
    "production": (
        "configs/evaluation/held_in_pooled_10core_hybrid_count_v1.yaml"
    ),
}

_FROZEN_MASKING: dict[str, Any] = {
    "type": "mixed_expression_masking",
    "curriculum": "P+N+B",
    "rate": {
        "partial_gene": 0.2,
        "whole_node": 0.1,
        "spatial_block": 0.1,
    },
    "rates": {
        "partial_gene": 0.2,
        "whole_node": 0.1,
        "spatial_block": 0.1,
    },
    "post_warmup_probabilities": {
        "partial_gene": 0.6,
        "whole_node": 0.3,
        "spatial_block": 0.1,
    },
    "warmup_epochs": 10,
    "block_shape": "disk",
    "block_width_um": None,
    "mask_seed": 314159,
    "validation_replicates": 0,
    "test_replicates": 0,
    "mask_expression_only": True,
    "explicit_gene_mask_channel": True,
}

_COUNT_REPRESENTATION: dict[str, Any] = {
    "schema": "hybrid_raw_count_0_1_2_3_4_7_8_15_16_31_32plus_v1",
    "source_scale": "raw_biological_probe_counts",
    "num_output_states": 8,
    "mask_token_id": 8,
    "mask_token_is_output": False,
    "fixed_boundaries": True,
    "fit_required": False,
    "count_mapping": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4-7": 4,
        "8-15": 5,
        "16-31": 6,
        "32+": 7,
    },
    "continuous_channel": {
        "source": "raw_biological_probe_counts",
        "transform": "shared_equal_core_standardized_log1p",
        "preserves_exact_within_bin_value": True,
        "masked_value": 0.0,
    },
}

_PROHIBITED_KEYS = frozenset(
    {
        "cell_id",
        "core_id",
        "core_label",
        "donor_id",
        "fov",
        "patient_id",
        "slide",
        "slide_id",
    }
)


class PooledHybridMaterializationError(ValueError):
    """Raised when a pooled campaign materialization fails closed."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PooledHybridMaterializationError(f"{label} must be a mapping")
    return value


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


def _yaml_bytes(value: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(value), sort_keys=False, allow_unicode=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, ValueError) as exc:
        raise PooledHybridMaterializationError(
            f"{label} is not strict readable JSON"
        ) from exc
    return dict(_mapping(payload, label))


def _project_reference(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise PooledHybridMaterializationError(
            "campaign path escapes the project root"
        ) from exc


def _resolve_reference(
    project_root: Path,
    value: str,
    *,
    label: str,
    kind: str,
) -> Path:
    reference = Path(value)
    if reference.is_absolute() or ".." in reference.parts:
        raise PooledHybridMaterializationError(
            f"{label} must be a safe project-relative path"
        )
    unresolved = project_root / reference
    if unresolved.is_symlink():
        raise PooledHybridMaterializationError(f"{label} cannot be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise PooledHybridMaterializationError(
            f"{label} escapes the project root"
        ) from exc
    if kind == "file" and not resolved.is_file():
        raise PooledHybridMaterializationError(f"{label} is not a file")
    if kind == "directory" and not resolved.is_dir():
        raise PooledHybridMaterializationError(f"{label} is not a directory")
    return resolved


def _component_section(
    project_root: Path, reference: str, section: str
) -> tuple[dict[str, Any], str]:
    path = _resolve_reference(
        project_root, reference, label=f"{section} component", kind="file"
    )
    payload = load_yaml_mapping(path)
    if set(payload) != {section}:
        raise PooledHybridMaterializationError(
            f"{reference} must contain only the {section!r} group"
        )
    return dict(_mapping(payload[section], f"{section} component")), sha256_file(
        path
    )


def _assert_alias_safe(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            canonical = str(key).lower().replace("-", "_")
            if canonical in _PROHIBITED_KEYS:
                raise PooledHybridMaterializationError(
                    f"{label} contains prohibited identifier field {key!r}"
                )
            _assert_alias_safe(child, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _assert_alias_safe(child, label=label)
    elif isinstance(value, str) and "SO_" in value:
        raise PooledHybridMaterializationError(
            f"{label} contains a prohibited source identifier"
        )


def _validate_frozen_contract(path: Path) -> tuple[dict[str, Any], str]:
    observed = sha256_file(path)
    if observed != EXPECTED_FROZEN_CONTRACT_SHA256:
        raise PooledHybridMaterializationError(
            "frozen pooled task contract differs from its locked SHA-256"
        )
    contract = load_yaml_mapping(path)
    if contract.get("campaign_id") != CAMPAIGN_ID:
        raise PooledHybridMaterializationError("frozen campaign ID drifted")
    cohort = _mapping(contract.get("cohort"), "contract.cohort")
    if (
        tuple(cohort.get("aliases", ())) != ALIASES
        or cohort.get("total_fit_cells") != EXPECTED_TOTAL_NODES
        or cohort.get("genes") != EXPECTED_N_GENES
    ):
        raise PooledHybridMaterializationError("frozen pooled cohort drifted")
    training = _mapping(contract.get("training"), "contract.training")
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "global_epochs": 200,
        "optimizer_steps_per_global_epoch": 10,
        "total_optimizer_steps": 2000,
        "core_order_seed": 271828,
        "mask_seed": 314159,
        "model_seeds": list(MODEL_SEEDS),
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise PooledHybridMaterializationError(
                f"frozen training field {key!r} drifted"
            )
    execution = _mapping(contract.get("execution"), "contract.execution")
    if (
        tuple(execution.get("safe_gpu_ids", ())) != SAFE_GPU_IDS
        or tuple(execution.get("pilot_gpu_ids", ())) != (0, 1)
        or tuple(execution.get("excluded_gpu_ids", ())) != (4,)
    ):
        raise PooledHybridMaterializationError("frozen GPU plan drifted")
    contract_model_names = {
        "pooled-hybrid-gat-k1000": "hybrid-gat-k1000",
        "pooled-hybrid-matched-self": "hybrid-matched-self",
    }
    for arm in ARMS:
        model = _mapping(
            _mapping(contract.get("models"), "contract.models").get(
                contract_model_names[arm]
            ),
            f"contract.models.{arm}",
        )
        if model.get("expected_trainable_parameters") != EXPECTED_PARAMETER_COUNT:
            raise PooledHybridMaterializationError(
                f"frozen parameter count drifted for {arm}"
            )
    return contract, observed


def _validate_prior_materialization(
    path: Path,
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    payload = _read_json(path, label="prior materialization")
    checksum = payload.get("checksum")
    core = dict(payload)
    core.pop("checksum", None)
    if (
        checksum != EXPECTED_PRIOR_MATERIALIZATION_CHECKSUM
        or canonical_sha256(core) != checksum
        or payload.get("campaign_id") != PRIOR_CAMPAIGN_ID
        or payload.get("parameter_count") != EXPECTED_PARAMETER_COUNT
    ):
        raise PooledHybridMaterializationError(
            "prior verified materialization identity does not match"
        )
    records = payload.get("cores")
    if not isinstance(records, list) or tuple(
        str(record.get("alias")) for record in records
    ) != ALIASES:
        raise PooledHybridMaterializationError(
            "prior materialization aliases are not exact and ordered"
        )
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        alias = str(record["alias"])
        graph_sha = record.get("k1000_graph_sha256")
        edges = record.get("k1000_directed_edges")
        if (
            record.get("n_genes") != EXPECTED_N_GENES
            or not isinstance(graph_sha, str)
            or len(graph_sha) != 64
            or isinstance(edges, bool)
            or not isinstance(edges, int)
            or edges <= 0
        ):
            raise PooledHybridMaterializationError(
                f"{alias} prior graph identity is invalid"
            )
        result[alias] = {
            "n_nodes": int(record["n_nodes"]),
            "n_genes": EXPECTED_N_GENES,
            "graph_sha256": graph_sha,
            "n_directed_edges": edges,
            "source_prepared_data_sha256": str(
                record["source_prepared_data_sha256"]
            ),
            "source_full_core_preprocessing_sha256": str(
                record["preprocessing_sha256"]
            ),
        }
    if sum(item["n_nodes"] for item in result.values()) != EXPECTED_TOTAL_NODES:
        raise PooledHybridMaterializationError(
            "prior materialization cell total is not 117386"
        )
    return payload, str(checksum), result


def _prepared_paths(
    project_root: Path, dataset_component: Mapping[str, Any]
) -> tuple[tuple[str, Path], ...]:
    if (
        tuple(dataset_component.get("core_aliases", ())) != ALIASES
        or dataset_component.get("total_fit_cells") != EXPECTED_TOTAL_NODES
        or dataset_component.get("biological_target_count") != EXPECTED_N_GENES
    ):
        raise PooledHybridMaterializationError(
            "pooled dataset component identity drifted"
        )
    raw = _mapping(
        dataset_component.get("prepared_artifacts"),
        "dataset.prepared_artifacts",
    )
    if tuple(raw) != ALIASES:
        raise PooledHybridMaterializationError(
            "prepared artifact alias map is not exact and ordered"
        )
    expected = {
        alias: (
            "data/processed/adjacent_normal_10core_qkv_large_k_v1/"
            f"{alias.lower()}/prepared_v1"
        )
        for alias in ALIASES
    }
    if dict(raw) != expected:
        raise PooledHybridMaterializationError(
            "prepared artifact path map differs from the frozen paths"
        )
    return tuple(
        (
            alias,
            _resolve_reference(
                project_root,
                expected[alias],
                label=f"{alias} prepared artifact",
                kind="directory",
            ),
        )
        for alias in ALIASES
    )


def _load_pooled_identity(
    *,
    project_root: Path,
    dataset_component: Mapping[str, Any],
    graph_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    paths = _prepared_paths(project_root, dataset_component)
    cohort = load_pooled_full_core_cohort(paths)
    if (
        tuple(cohort.aliases) != ALIASES
        or cohort.total_nodes != EXPECTED_TOTAL_NODES
        or cohort.n_genes != EXPECTED_N_GENES
        or tuple(cohort.metadata_names) != tuple(ALLOWED_METADATA_COLUMNS)
        or len(cohort.gene_names) != EXPECTED_N_GENES
        or len(cohort.fingerprint_sha256) != 64
    ):
        raise PooledHybridMaterializationError(
            "loaded pooled cohort violates its frozen schema"
        )
    path_map = {
        alias: _project_reference(project_root, path) for alias, path in paths
    }
    core_records: dict[str, dict[str, Any]] = {}
    for loaded, receipt in zip(
        cohort.cores, EXPECTED_PREPARED_CORES, strict=True
    ):
        source = graph_sources[loaded.alias]
        if (
            loaded.alias != receipt.alias
            or loaded.n_nodes != source["n_nodes"]
            or loaded.n_genes != EXPECTED_N_GENES
            or loaded.checksums.source_prepared_data_sha256
            != source["source_prepared_data_sha256"]
            or loaded.checksums.source_full_core_preprocessing_sha256
            != source["source_full_core_preprocessing_sha256"]
            or loaded.preprocessing_qc.protected_identifier_arrays_returned
            is not False
        ):
            raise PooledHybridMaterializationError(
                f"{loaded.alias} pooled/source identity mismatch"
            )
        core_records[loaded.alias] = {
            "n_nodes": loaded.n_nodes,
            "n_genes": loaded.n_genes,
            "prepared_artifact": path_map[loaded.alias],
            "source_manifest_sha256": (
                loaded.checksums.source_manifest_sha256
            ),
            "source_prepared_data_sha256": (
                loaded.checksums.source_prepared_data_sha256
            ),
            "source_full_core_preprocessing_sha256": (
                loaded.checksums.source_full_core_preprocessing_sha256
            ),
            "pooled_preprocessing_sha256": (
                loaded.checksums.preprocessing_sha256
            ),
        }
    split_basis = {
        "schema": "held_in_pooled_10core_all_fit_roles_v1",
        "dataset_fingerprint": cohort.fingerprint_sha256,
        "aliases": list(ALIASES),
        "total_nodes": EXPECTED_TOTAL_NODES,
        "role_assignment": "all rows assigned fit within each core",
        "role_counts": {
            "fit": EXPECTED_TOTAL_NODES,
            "validation": 0,
            "test": 0,
        },
        "experimental_unit": "adjacent_normal_spatial_core",
    }
    split_fingerprint = canonical_sha256(split_basis)
    result = {
        "aliases": list(ALIASES),
        "total_nodes": cohort.total_nodes,
        "n_genes": cohort.n_genes,
        "dataset_fingerprint": cohort.fingerprint_sha256,
        "cohort_checksums": cohort.checksums.to_dict(),
        "split_id": split_fingerprint[:16],
        "split_fingerprint": split_fingerprint,
        "split_fingerprint_basis": split_basis,
        "prepared_artifacts": path_map,
        "cores": core_records,
    }
    del cohort
    gc.collect()
    return result


def _default_prior_mask_sources(project_root: Path) -> dict[str, Path]:
    return {
        alias: (
            project_root
            / "artifacts"
            / "runs"
            / run_id[2:6]
            / run_id[6:8]
            / run_id
        )
        for alias, run_id in _PRIOR_GAT_RUN_IDS.items()
    }


def _validate_prior_mask_sources(
    *,
    project_root: Path,
    graph_sources: Mapping[str, Mapping[str, Any]],
    prior_mask_sources: Mapping[str, Path] | None,
) -> dict[str, dict[str, Any]]:
    sources = (
        _default_prior_mask_sources(project_root)
        if prior_mask_sources is None
        else dict(prior_mask_sources)
    )
    if tuple(sources) != ALIASES:
        raise PooledHybridMaterializationError(
            "prior mask source aliases are not exact and ordered"
        )
    identities: dict[str, dict[str, Any]] = {}
    for alias in ALIASES:
        root = Path(sources[alias]).resolve()
        try:
            root.relative_to(project_root)
        except ValueError as exc:
            raise PooledHybridMaterializationError(
                f"{alias} prior mask source escapes the project root"
            ) from exc
        verification = verify_run_bundle(root, require_success_contract=True)
        if verification.get("status") != "success":
            raise PooledHybridMaterializationError(
                f"{alias} prior mask source is not a verified success"
            )
        config = load_yaml_mapping(root / "config.resolved.yaml")
        summary = _read_json(root / "summary.json", label=f"{alias} summary")
        masks_path = root / "provenance" / "fixed_evaluation_masks.json"
        masks = _read_json(masks_path, label=f"{alias} fixed masks")
        experiment = _mapping(config.get("experiment"), f"{alias} experiment")
        model = _mapping(config.get("model"), f"{alias} model")
        if (
            summary.get("campaign_id") != PRIOR_CAMPAIGN_ID
            or summary.get("biological_unit_alias") != alias
            or summary.get("model_seed") != 0
            or summary.get("model_name") != "hybrid-count-gat"
            or summary.get("graph_sha256")
            != graph_sources[alias]["graph_sha256"]
            or summary.get("graph_directed_edges")
            != graph_sources[alias]["n_directed_edges"]
            or config.get("seed") != 0
            or experiment.get("arm") != "hybrid-gat-k1000"
            or model.get("name") != "hybrid-count-gat"
        ):
            raise PooledHybridMaterializationError(
                f"{alias} prior GAT mask source identity is inconsistent"
            )
        manifest = dict(
            _mapping(masks.get("bundle_manifest"), f"{alias} mask manifest")
        )
        checksum = manifest.get("bundle_checksum")
        checksum_core = deepcopy(manifest)
        checksum_core.pop("bundle_checksum", None)
        checksum_core.pop("bundle_id", None)
        entries = manifest.get("entries")
        if (
            not isinstance(checksum, str)
            or canonical_sha256(checksum_core) != checksum
            or summary.get("evaluation_mask_bundle_sha256") != checksum
            or manifest.get("bundle_id") != checksum[:16]
            or manifest.get("n_genes") != EXPECTED_N_GENES
            or manifest.get("replicates") != 3
            or not isinstance(entries, list)
            or len(entries) != 9
        ):
            raise PooledHybridMaterializationError(
                f"{alias} prior mask manifest does not verify"
            )
        compact_entries: list[dict[str, Any]] = []
        slots: set[tuple[str, int]] = set()
        for raw_entry in entries:
            entry = _mapping(raw_entry, f"{alias} mask entry")
            spec = _mapping(entry.get("spec"), f"{alias} mask spec")
            mode = str(spec.get("label"))
            replicate = entry.get("replicate")
            seed = entry.get("seed")
            mask_checksum = entry.get("mask_checksum")
            if (
                mode
                not in {"partial_gene", "whole_node", "spatial_block"}
                or replicate not in {0, 1, 2}
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or not isinstance(mask_checksum, str)
                or len(mask_checksum) != 64
                or entry.get("shape")
                != [graph_sources[alias]["n_nodes"], EXPECTED_N_GENES]
            ):
                raise PooledHybridMaterializationError(
                    f"{alias} prior mask seed/checksum identity is invalid"
                )
            slots.add((mode, int(replicate)))
            compact_entries.append(
                {
                    "entry_id": str(entry["entry_id"]),
                    "mode": mode,
                    "replicate": int(replicate),
                    "seed": seed,
                    "mask_checksum": mask_checksum,
                }
            )
        expected_slots = {
            (mode, replicate)
            for mode in ("partial_gene", "whole_node", "spatial_block")
            for replicate in range(3)
        }
        if slots != expected_slots:
            raise PooledHybridMaterializationError(
                f"{alias} prior mask slots are incomplete or duplicated"
            )
        identities[alias] = {
            "source_run_id": root.name,
            "reference": _project_reference(project_root, masks_path),
            "file_sha256": sha256_file(masks_path),
            "bundle_checksum": checksum,
            "base_seed": int(manifest["base_seed"]),
            "entries": compact_entries,
        }
    return identities


def _validate_model_components(
    model_sections: Mapping[str, Mapping[str, Any]]
) -> int:
    graph = model_sections["pooled-hybrid-gat-k1000"]
    self_only = model_sections["pooled-hybrid-matched-self"]
    expected_graph = {
        "name": "hybrid-count-gat",
        "family": "hybrid_count_edge_conditioned_gatv2",
        "embedding_dim": 512,
        "hidden_dim": 512,
        "graph_layers": 2,
        "attention_heads": 4,
        "ffn_dim": 512,
        "decoder_dim": 512,
        "uses_graph_inputs": True,
        "uses_edge_inputs": True,
    }
    for key, expected in expected_graph.items():
        if graph.get(key) != expected:
            raise PooledHybridMaterializationError(
                f"hybrid GAT model field {key!r} drifted"
            )
    for key, expected in {
        **expected_graph,
        "name": "hybrid-count-matched-self",
        "family": "hybrid_count_parameter_matched_self_control",
        "uses_graph_inputs": False,
        "uses_edge_inputs": False,
    }.items():
        if self_only.get(key) != expected:
            raise PooledHybridMaterializationError(
                f"hybrid self model field {key!r} drifted"
            )
    paired_fields = (
        "embedding_dim",
        "hidden_dim",
        "graph_layers",
        "attention_heads",
        "ffn_dim",
        "decoder_dim",
        "edge_hidden_dim",
        "edge_embedding_dim",
        "dropout",
        "attention_dropout",
    )
    if any(graph.get(key) != self_only.get(key) for key in paired_fields):
        raise PooledHybridMaterializationError(
            "hybrid model component dimensions do not match"
        )
    common = {
        "num_genes": EXPECTED_N_GENES,
        "edge_attribute_dim": len(EDGE_ATTRIBUTE_NAMES),
        "expression_mean": np.zeros(EXPECTED_N_GENES, dtype=np.float32),
        "expression_scale": np.ones(EXPECTED_N_GENES, dtype=np.float32),
        "node_covariate_dim": len(ALLOWED_METADATA_COLUMNS),
        "hidden_dim": int(graph["hidden_dim"]),
        "attention_heads": int(graph["attention_heads"]),
        "attention_head_dim": graph.get("attention_head_dim"),
        "graph_layers": int(graph["graph_layers"]),
        "ffn_dim": int(graph["ffn_dim"]),
        "decoder_dim": int(graph["decoder_dim"]),
        "edge_hidden_dim": int(graph["edge_hidden_dim"]),
        "edge_embedding_dim": int(graph["edge_embedding_dim"]),
        "dropout": float(graph["dropout"]),
        "attention_dropout": float(graph["attention_dropout"]),
    }
    graph_model = HybridReceiverChunkedEdgeConditionedGATv2(
        **common,
        receiver_chunk_size=int(graph["receiver_chunk_size"]),
        activation_checkpointing=bool(graph["activation_checkpointing"]),
    )
    self_model = HybridEdgeParameterMatchedSelfControl(**common)
    count = assert_exact_parameter_match(graph_model, self_model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise PooledHybridMaterializationError(
            "hybrid arms do not have the frozen parameter count"
        )
    if (
        graph_model.encoder.__class__ is not self_model.encoder.__class__
        or graph_model.decoder.__class__ is not self_model.decoder.__class__
    ):
        raise PooledHybridMaterializationError(
            "paired encoder or decoder classes differ"
        )
    try:
        self_model.encoder.load_state_dict(
            graph_model.encoder.state_dict(), strict=True
        )
        self_model.decoder.load_state_dict(
            graph_model.decoder.state_dict(), strict=True
        )
    except RuntimeError as exc:
        raise PooledHybridMaterializationError(
            "paired common initialization cannot be copied exactly"
        ) from exc
    for label, graph_module, self_module in (
        ("encoder", graph_model.encoder, self_model.encoder),
        ("decoder", graph_model.decoder, self_model.decoder),
    ):
        graph_state = graph_module.state_dict()
        self_state = self_module.state_dict()
        if list(graph_state) != list(self_state) or any(
            graph_state[key].shape != self_state[key].shape
            or graph_state[key].dtype != self_state[key].dtype
            or not torch.equal(graph_state[key], self_state[key])
            for key in graph_state
        ):
            raise PooledHybridMaterializationError(
                f"paired {label} initialization is not bit-identical"
            )
    del graph_model, self_model
    gc.collect()
    return count


def _validate_components(
    *,
    dataset: Mapping[str, Any],
    features: Mapping[str, Any],
    graph: Mapping[str, Any],
    trainers: Mapping[str, Mapping[str, Any]],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> None:
    if (
        dataset.get("fit_scope")
        != "all_117386_cells_across_ten_cores_transductive"
        or graph.get("neighbor_k") != 1000
        or graph.get("k") != 1000
        or graph.get("symmetry") != "mutual"
        or graph.get("edge_dropout") != 0.0
        or graph.get("cross_core_edges") is not False
        or tuple(
            _mapping(features.get("node_metadata"), "features.node_metadata").get(
                "fields", ()
            )
        )
        != tuple(ALLOWED_METADATA_COLUMNS)
        or tuple(
            _mapping(features.get("edge_features"), "features.edge_features").get(
                "fields", ()
            )
        )
        != tuple(EDGE_ATTRIBUTE_NAMES)
    ):
        raise PooledHybridMaterializationError(
            "pooled data, feature, or graph component drifted"
        )
    for role in ("pilot", "production"):
        trainer = trainers[role]
        evaluation = evaluations[role]
        expected_epochs = 2 if role == "pilot" else 200
        expected_steps = 20 if role == "pilot" else 2000
        if (
            trainer.get("optimizer") != "AdamW"
            or trainer.get("learning_rate") != 3e-4
            or trainer.get("weight_decay") != 1e-4
            or trainer.get("gradient_clip_norm") != 1.0
            or trainer.get("max_epochs") != expected_epochs
            or trainer.get("total_optimizer_steps") != expected_steps
            or trainer.get("core_order_seed") != 271828
            or trainer.get("neighbor_sampling") is not False
            or trainer.get("early_stopping") is not False
            or trainer.get("checkpoint_policy") != "last_only"
            or evaluation.get("protocol")
            != "held_in_pooled_10core_fixed_budget"
            or evaluation.get("mask_replicates_per_mode") != 3
            or evaluation.get("mask_source")
            != "exact_regeneration_of_prior_per_core_fixed_masks"
        ):
            raise PooledHybridMaterializationError(
                f"pooled {role} trainer/evaluation component drifted"
            )


def _dataset_section(
    component: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    dataset = deepcopy(dict(component))
    dataset.update(
        {
            "split_id": identity["split_id"],
            "dataset_fingerprint": identity["dataset_fingerprint"],
            "dataset_fingerprint_role": (
                "materialized_pooled_ten_core_preprocessing_checksum"
            ),
            "dataset_fingerprint_basis": {
                "schema": "pooled_full_core_cohort_v1",
                "cohort_checksums": deepcopy(identity["cohort_checksums"]),
                "prepared_artifacts": deepcopy(identity["prepared_artifacts"]),
                "core_content_sha256": {
                    alias: {
                        "source_manifest_sha256": record[
                            "source_manifest_sha256"
                        ],
                        "source_prepared_data_sha256": record[
                            "source_prepared_data_sha256"
                        ],
                        "pooled_preprocessing_sha256": record[
                            "pooled_preprocessing_sha256"
                        ],
                    }
                    for alias, record in identity["cores"].items()
                },
                "expression_transform": (
                    "equal_core_mixture_shared_gene_wise_standardized_log1p"
                ),
                "morphology_transform": (
                    "existing_per_core_all_fit_standardization"
                ),
                "fit_scope": "all_117386_cells_transductive",
            },
            "split_fingerprint": identity["split_fingerprint"],
            "split_fingerprint_status": (
                "verified_materialized_no_holdout_role_assignment"
            ),
            "split_fingerprint_basis": deepcopy(
                identity["split_fingerprint_basis"]
            ),
            "prepared_artifacts": deepcopy(identity["prepared_artifacts"]),
            "prepared_artifact_sha256": {
                alias: {
                    "source_manifest_sha256": record[
                        "source_manifest_sha256"
                    ],
                    "source_prepared_data_sha256": record[
                        "source_prepared_data_sha256"
                    ],
                    "pooled_preprocessing_sha256": record[
                        "pooled_preprocessing_sha256"
                    ],
                }
                for alias, record in identity["cores"].items()
            },
            "core_node_counts": {
                alias: record["n_nodes"]
                for alias, record in identity["cores"].items()
            },
            "count_representation": deepcopy(_COUNT_REPRESENTATION),
        }
    )
    return dataset


def _features_section(
    component: Mapping[str, Any], *, uses_graph: bool
) -> dict[str, Any]:
    features = deepcopy(dict(component))
    features["use_edge_features"] = uses_graph
    features["node_expression"] = {
        "biological_targets": EXPECTED_N_GENES,
        "source_scale": "raw_biological_probe_counts",
        "discrete_transform": "fixed_hybrid_count_states",
        "continuous_transform": "shared_equal_core_standardized_log1p",
        "masked_discrete_value": "input_only_mask_token_8",
        "masked_continuous_value": 0.0,
        "explicit_mask_authoritative_inside_model": True,
    }
    if not uses_graph:
        features["edge_features"] = []
    prohibited = list(features.get("prohibited_node_inputs", ()))
    for value in (
        "core_alias",
        "rna_derived_qc",
        "hidden_target_values",
    ):
        if value not in prohibited:
            prohibited.append(value)
    features["prohibited_node_inputs"] = prohibited
    return features


def _graph_section(
    component: Mapping[str, Any],
    graph_sources: Mapping[str, Mapping[str, Any]],
    *,
    uses_graph: bool,
) -> dict[str, Any]:
    graph = deepcopy(dict(component))
    graph["model_graph_input_enabled"] = uses_graph
    graph["expected_core_graphs"] = {
        alias: {
            "n_nodes": source["n_nodes"],
            "graph_sha256": source["graph_sha256"],
            "n_directed_edges": source["n_directed_edges"],
        }
        for alias, source in graph_sources.items()
    }
    return graph


def _experiment_section(*, arm: str, pilot: bool) -> dict[str, Any]:
    arm_key = arm.replace("-", "_")
    return {
        "variant_label": (
            f"pooled_{arm_key}_resource_pilot"
            if pilot
            else f"pooled_{arm_key}_ensemble_member"
        ),
        "arm": arm,
        "core_aliases": list(ALIASES),
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "estimand": (
            "pooled_implementation_resource_feasibility"
            if pilot
            else "held_in_pooled_ten_core_masked_expression"
        ),
        "permitted_claim": (
            "diagnostic_runtime_memory_and_precision_only"
            if pilot
            else (
                "held_in_shared_representation_capacity_and_descriptive_"
                "broad_context_graph_gain"
            )
        ),
        "one_shared_model_state": True,
        "cross_core_edges": False,
        "conclusion_eligible": not pilot,
        "excluded_from_primary_comparison": pilot,
        "resource_pilot": pilot,
    }


def _classification_section(*, arm: str, pilot: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "lifecycle_stage": "diagnostic" if pilot else "exploratory_screen",
        "study_axis": (
            "pooled_hybrid_count_resource_feasibility"
            if pilot
            else "pooled_hybrid_count_ensemble_graph_gain"
        ),
        "scientific_variant": arm,
        "retention_class": (
            "retain_diagnostic_evidence"
            if pilot
            else "retain_exploratory_evidence"
        ),
        "classification_confidence": "high",
    }


def _build_config(
    *,
    arm: str,
    seed: int,
    pilot: bool,
    requested_gpu: int,
    output_reference: str,
    contract_reference: str,
    contract_sha256: str,
    prior_materialization_reference: str,
    prior_materialization_checksum: str,
    dataset_component: Mapping[str, Any],
    features_component: Mapping[str, Any],
    graph_component: Mapping[str, Any],
    model_sections: Mapping[str, Mapping[str, Any]],
    trainer_sections: Mapping[str, Mapping[str, Any]],
    evaluation_sections: Mapping[str, Mapping[str, Any]],
    launcher_component: Mapping[str, Any],
    cohort_identity: Mapping[str, Any],
    graph_sources: Mapping[str, Mapping[str, Any]],
    mask_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    uses_graph = arm == "pooled-hybrid-gat-k1000"
    role = "pilot" if pilot else "production"
    trainer = deepcopy(dict(trainer_sections[role]))
    gate_reference = f"{output_reference}/{PILOT_GATE_RECEIPT_NAME}"
    trainer["amp_authorization"] = {
        "mode": (
            "same_weight_fp32_amp_each_core_diagnostic"
            if pilot
            else "require_external_pilot_gate_receipt"
        ),
        "receipt_schema": PILOT_GATE_RECEIPT_KIND,
        "receipt_reference": gate_reference,
        "frozen_contract_sha256": contract_sha256,
    }
    evaluation = deepcopy(dict(evaluation_sections[role]))
    evaluation["prior_mask_sources"] = deepcopy(dict(mask_sources))
    evaluation["mask_seed_namespace"] = (
        "exact_prior_held_in_full_core_fixed_evaluation"
    )
    launcher = deepcopy(dict(launcher_component))
    launcher["requested_gpu"] = str(requested_gpu)
    config: dict[str, Any] = {
        "model": deepcopy(dict(model_sections[arm])),
        "masking": deepcopy(_FROZEN_MASKING),
        "dataset": _dataset_section(dataset_component, cohort_identity),
        "features": _features_section(
            features_component, uses_graph=uses_graph
        ),
        "graph": _graph_section(
            graph_component, graph_sources, uses_graph=uses_graph
        ),
        "trainer": trainer,
        "evaluation": evaluation,
        "launcher": launcher,
        "version": 1,
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "display_name": (
                "Pooled ten-core adjacent-normal hybrid-count ensemble"
            ),
            "exploratory": True,
            "frozen_contract": contract_reference,
            "frozen_contract_sha256": contract_sha256,
        },
        "experiment": _experiment_section(arm=arm, pilot=pilot),
        "classification": _classification_section(arm=arm, pilot=pilot),
        "metadata": {
            "locked_config_materialization_receipt": (
                f"{output_reference}/{RECEIPT_NAME}"
            ),
            "frozen_scientific_contract": True,
            "execution_role": "resource_pilot" if pilot else "production",
            "production_requires_pilot_gate": not pilot,
            "pilot_gate_receipt_reference": gate_reference,
            "prior_materialization_reference": (
                prior_materialization_reference
            ),
            "prior_materialization_checksum": (
                prior_materialization_checksum
            ),
            "paired_common_initialization_required": True,
        },
        "seed": seed,
        "fold": 0,
        "attempt": 1,
    }
    validate_experiment_config(config)
    command = command_for_config(config)
    if (
        not command
        or not any("run_pooled_hybrid_count_capacity.py" in part for part in command)
    ):
        raise PooledHybridMaterializationError(
            "resolved pooled config does not derive the pooled runner command"
        )
    _assert_alias_safe(config, label="resolved pooled config")
    if requested_gpu not in SAFE_GPU_IDS or requested_gpu == 4:
        raise PooledHybridMaterializationError(
            "resolved pooled config requests an unsafe GPU"
        )
    return config


def _config_filename(*, arm: str, seed: int, pilot: bool) -> str:
    role = "_resource_pilot" if pilot else ""
    return f"seed-{seed:02d}_{arm.replace('-', '_')}{role}.yaml"


def _existing_output_matches(
    output_dir: Path, expected_files: Mapping[str, bytes]
) -> bool:
    if not output_dir.is_dir() or output_dir.is_symlink():
        return False
    observed = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if observed != set(expected_files):
        return False
    return all(
        (output_dir / relative).read_bytes() == content
        for relative, content in expected_files.items()
    )


def _publish_atomically(
    output_dir: Path, files: Mapping[str, bytes]
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.stage-", dir=output_dir.parent
        )
    )
    try:
        for relative, content in files.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        if not _existing_output_matches(stage, files):
            raise PooledHybridMaterializationError(
                "staged pooled materialization failed byte verification"
            )
        if output_dir.exists() or output_dir.is_symlink():
            if _existing_output_matches(output_dir, files):
                return
            raise PooledHybridMaterializationError(
                "locked output exists with different content"
            )
        os.replace(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def materialize_campaign(
    *,
    project_root: Path,
    prior_materialization: Path,
    contract_path: Path,
    output_dir: Path,
    prior_mask_sources: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish all sixteen locked pooled configs."""

    project_root = project_root.resolve()
    prior_materialization = prior_materialization.resolve()
    contract_path = contract_path.resolve()
    output_dir = output_dir.resolve()
    for path, label in (
        (prior_materialization, "prior materialization"),
        (contract_path, "frozen task contract"),
        (output_dir, "output directory"),
    ):
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise PooledHybridMaterializationError(
                f"{label} must remain under the project root"
            ) from exc
    if output_dir == project_root:
        raise PooledHybridMaterializationError("output directory is too broad")

    _contract, contract_sha = _validate_frozen_contract(contract_path)
    (
        _prior,
        prior_checksum,
        graph_sources,
    ) = _validate_prior_materialization(prior_materialization)

    sections: dict[str, Mapping[str, Any]] = {}
    component_hashes: dict[str, str] = {}
    for section, reference in _COMPONENTS.items():
        body, checksum = _component_section(
            project_root, reference, section
        )
        sections[section] = body
        component_hashes[reference] = checksum
    models: dict[str, Mapping[str, Any]] = {}
    for arm, reference in _MODEL_COMPONENTS.items():
        body, checksum = _component_section(project_root, reference, "model")
        models[arm] = body
        component_hashes[reference] = checksum
    trainers: dict[str, Mapping[str, Any]] = {}
    evaluations: dict[str, Mapping[str, Any]] = {}
    for role in ("pilot", "production"):
        reference = _TRAINER_COMPONENTS[role]
        trainers[role], component_hashes[reference] = _component_section(
            project_root, reference, "trainer"
        )
        reference = _EVALUATION_COMPONENTS[role]
        evaluations[role], component_hashes[reference] = _component_section(
            project_root, reference, "evaluation"
        )

    _validate_components(
        dataset=sections["dataset"],
        features=sections["features"],
        graph=sections["graph"],
        trainers=trainers,
        evaluations=evaluations,
    )
    parameter_count = _validate_model_components(models)
    cohort_identity = _load_pooled_identity(
        project_root=project_root,
        dataset_component=sections["dataset"],
        graph_sources=graph_sources,
    )
    mask_sources = _validate_prior_mask_sources(
        project_root=project_root,
        graph_sources=graph_sources,
        prior_mask_sources=prior_mask_sources,
    )

    output_reference = _project_reference(project_root, output_dir)
    contract_reference = _project_reference(project_root, contract_path)
    prior_reference = _project_reference(
        project_root, prior_materialization
    )
    generated: dict[str, bytes] = {}
    pilot_jobs: list[dict[str, Any]] = []
    production_jobs: list[dict[str, Any]] = []
    plans = (
        (
            True,
            [
                (arm, 0, PILOT_GPU_MAP[arm])
                for arm in ARMS
            ],
            "resource_pilot_configs",
            pilot_jobs,
        ),
        (
            False,
            [
                (arm, seed, SEED_GPU_MAP[seed])
                for seed in MODEL_SEEDS
                for arm in ARMS
            ],
            "production_configs",
            production_jobs,
        ),
    )
    for pilot, jobs, directory, records in plans:
        for arm, seed, requested_gpu in jobs:
            config = _build_config(
                arm=arm,
                seed=seed,
                pilot=pilot,
                requested_gpu=requested_gpu,
                output_reference=output_reference,
                contract_reference=contract_reference,
                contract_sha256=contract_sha,
                prior_materialization_reference=prior_reference,
                prior_materialization_checksum=prior_checksum,
                dataset_component=sections["dataset"],
                features_component=sections["features"],
                graph_component=sections["graph"],
                model_sections=models,
                trainer_sections=trainers,
                evaluation_sections=evaluations,
                launcher_component=sections["launcher"],
                cohort_identity=cohort_identity,
                graph_sources=graph_sources,
                mask_sources=mask_sources,
            )
            relative = (
                f"{directory}/"
                f"{_config_filename(arm=arm, seed=seed, pilot=pilot)}"
            )
            content = _yaml_bytes(config)
            generated[relative] = content
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "config": f"{output_reference}/{relative}",
                    "config_sha256": canonical_sha256(config),
                    "file_sha256": _sha256_bytes(content),
                    "requested_gpu": requested_gpu,
                }
            )

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": RECEIPT_KIND,
        "campaign_id": CAMPAIGN_ID,
        "frozen_contract": {
            "reference": contract_reference,
            "sha256": contract_sha,
        },
        "prior_materialization": {
            "campaign_id": PRIOR_CAMPAIGN_ID,
            "reference": prior_reference,
            "file_sha256": sha256_file(prior_materialization),
            "canonical_checksum": prior_checksum,
        },
        "component_file_sha256": dict(sorted(component_hashes.items())),
        "parameter_count": parameter_count,
        "parameter_counts": {
            arm: parameter_count for arm in ARMS
        },
        "paired_common_initialization_required": True,
        "cohort": deepcopy(cohort_identity),
        "cores": [
            {
                "alias": alias,
                **deepcopy(cohort_identity["cores"][alias]),
                "graph_sha256": graph_sources[alias]["graph_sha256"],
                "n_directed_edges": graph_sources[alias][
                    "n_directed_edges"
                ],
            }
            for alias in ALIASES
        ],
        "graph_sources": deepcopy(graph_sources),
        "evaluation_mask_sources": deepcopy(mask_sources),
        "allowed_gpu_ids": list(SAFE_GPU_IDS),
        "excluded_gpu_ids": [4],
        "assignment": {
            "policy": "fixed_seed_to_gpu_v1",
            "seed_gpu_map": {
                str(seed): gpu for seed, gpu in SEED_GPU_MAP.items()
            },
            "pilot_gpu_map": deepcopy(PILOT_GPU_MAP),
        },
        "pilot_jobs": pilot_jobs,
        "production_jobs": production_jobs,
        "pilot_gate_receipt_reference": (
            f"{output_reference}/{PILOT_GATE_RECEIPT_NAME}"
        ),
        "counts": {
            "aliases": 10,
            "pilot_configs": len(pilot_jobs),
            "production_configs": len(production_jobs),
            "production_seeds": len(MODEL_SEEDS),
        },
        "registry_mutation_performed": False,
        "queue_mutation_performed": False,
        "training_performed": False,
    }
    if receipt["counts"] != {
        "aliases": 10,
        "pilot_configs": 2,
        "production_configs": 14,
        "production_seeds": 7,
    }:
        raise PooledHybridMaterializationError(
            "resolved pooled config count is not exactly 2 pilot plus 14 production"
        )
    if {
        (job["arm"], job["seed"]) for job in production_jobs
    } != {
        (arm, seed) for arm in ARMS for seed in MODEL_SEEDS
    }:
        raise PooledHybridMaterializationError(
            "production arm/seed slots are not exact"
        )
    _assert_alias_safe(receipt, label="pooled materialization receipt")
    receipt["checksum"] = canonical_sha256(receipt)
    generated[RECEIPT_NAME] = _json_bytes(receipt)

    if output_dir.exists() or output_dir.is_symlink():
        if not _existing_output_matches(output_dir, generated):
            raise PooledHybridMaterializationError(
                "locked output exists with different content"
            )
        return receipt
    _publish_atomically(output_dir, generated)
    if not _existing_output_matches(output_dir, generated):
        raise PooledHybridMaterializationError(
            "published pooled materialization failed verification"
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-materialization",
        type=Path,
        default=(
            paths.scratch_root
            / "locked_campaigns"
            / PRIOR_CAMPAIGN_ID
            / RECEIPT_NAME
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            paths.project_root
            / "experiments"
            / "campaigns"
            / CAMPAIGN_ID
            / "frozen_task_contract.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = current_paths()
    receipt = materialize_campaign(
        project_root=paths.project_root,
        prior_materialization=args.prior_materialization,
        contract_path=args.contract,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "checksum": receipt["checksum"],
                "counts": receipt["counts"],
                "output": _project_reference(paths.project_root, args.output_dir),
                "queue_mutation_performed": False,
                "registry_mutation_performed": False,
                "training_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

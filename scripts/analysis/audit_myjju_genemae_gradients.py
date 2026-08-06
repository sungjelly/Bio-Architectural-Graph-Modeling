#!/usr/bin/env python3
"""Execute the frozen MyJJu GeneMAE gradient audit.

This entry point owns strict upstream discovery and transient artifact
publication.  The numerical estimands live in
``spatial_benchmark.myjju_gradient_audit``.  In particular, a summed-output
VJP is never labelled as a same-cell/other-cell decomposition: that split is
accepted only from exact receiver-resolved scalar gradients.

The workflow intentionally does not create, register, or finalize a run.
``pilot`` and ``seed-shard`` write only under the caller's active scratch
directory.  ``aggregate`` verifies those immutable inputs before publishing
aggregate reports.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import hashlib
import html
import importlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import load_yaml_mapping  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.myjju_gradient_audit import (  # noqa: E402
    FROZEN_LOCKED_DIRECTED_PAIRS,
    FROZEN_MARKER_GENES,
    FROZEN_PERTURBATION_SCALES,
    FROZEN_RECEIVER_SAMPLES_PER_TARGET_PER_TILE,
    FROZEN_SOURCE_SELECTED_PAIRS,
    FROZEN_TARGET_GENES,
    compare_central_bounded_perturbation,
    decomposed_masked_input_gradients,
    make_bounded_source_perturbation,
    summed_output_input_gradients,
)
from spatial_benchmark.myjju_genemae_comparison import (  # noqa: E402
    RegisteredRunEvidence,
    discover_registered_genemae_production,
)
from spatial_benchmark.myjju_gradient_reduction import (  # noqa: E402
    GradientReductionError,
    reduce_gradient_audit,
)
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


CAMPAIGN_ID = "cmp_20260731_myjju_genemae_gradient_audit_cpu_replay"
UPSTREAM_CAMPAIGN_ID = "cmp_20260730_myjju_genemae_10core_comparison"
FROZEN_CONTRACT_SHA256 = (
    "bcdd6d7223f9b97f26833f77935c75645110aaba1669d11c0a7180c957514cc1"
)
IMPLEMENTATION_PROTOCOL_SHA256 = (
    "b4328095bcde383633fb2862b3c35b6bc8203c98906aee7f2058443b2d16ddb8"
)
EXTERNAL_SOURCE_COMMIT = "f9ef61071c7e9b2751bbd59d154c13de534e7f2f"
EXTERNAL_SOURCE_SCRIPT_SHA256 = (
    "0ae143b5957e9275882ba595702d6eacd033545beda306f0f0f82905a8174680"
)
SOURCE_STATISTIC_AUDIT_SHA256 = (
    "f9b129f0b06d491d9ff9af0aa9b579ba1a384e1e958c0c9dd036e60348370bce"
)
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
SEEDS = tuple(range(7))
MASK_REPLICATES = (0, 1, 2)
GRAPH_CONDITIONS = ("observed", "node_label_permuted")
EXPECTED_TOTAL_NODES = 117_386
EXPECTED_TOTAL_TILES = 26
TILE_COUNTS = {
    "ANC-01": 4,
    "ANC-02": 2,
    "ANC-03": 4,
    "ANC-04": 2,
    "ANC-05": 2,
    "ANC-06": 2,
    "ANC-07": 2,
    "ANC-08": 4,
    "ANC-09": 2,
    "ANC-10": 2,
}
EXPECTED_GRAPH_SHA256 = (
    "f860f84f96e9daf4c9c4e4ddd5fc40aeacd3f6362c9e97d07248ac2575b462d5"
)
EXPECTED_COHORT_SHA256 = (
    "b3c06228ce7fda4d4c5da09c3281de46e67b8069dfadb14276ba6402af87767e"
)
RANDOM_MODEL_SEEDS = tuple(range(9100, 9107))
PRODUCTION_GPU_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7}
DEFAULT_WORK_RELATIVE = Path(
    "active_runs/posthoc_myjju_genemae_gradient_audit_v1"
)
DEFAULT_REPORT_RELATIVE = Path(
    "analyses/myjju_genemae_gradient_audit"
)
DATASET_ID = "cosmx_adjacent_normal_10core_pooled_fit_v1"
SHARD_KIND = "myjju_genemae_gradient_audit_seed_shard"
PILOT_KIND = "myjju_genemae_gradient_audit_resource_pilot"
MANIFEST_KIND = "myjju_genemae_gradient_audit_report_manifest"
ARRAY_RETENTION = "transient_deleted_after_verified_aggregation"
FINAL_METRIC_NAME = "audit/gate_row_pass_fraction_descriptive"

# NPZ arrays are transient and contain no row identifiers.  Selected-target
# predictions remain in verified scratch only until the nonlinear seven-model
# ensemble metrics have been recomputed.
REQUIRED_ARRAYS = {
    "source_unmasked_signed": (10, 39, 39),
    "masked_rep0_observed_signed": (10, 39, 39),
    "masked_rep0_permuted_signed": (10, 39, 39),
    "masked_locked_observed_signed": (10, 3, 5, 39),
    "decomposition_same_signed": (10, 5, 39),
    "decomposition_other_signed": (10, 5, 39),
    "decomposition_total_signed": (10, 5, 39),
    "decomposition_same_l1": (10, 5, 39),
    "decomposition_other_l1": (10, 5, 39),
    "decomposition_total_l1": (10, 5, 39),
    "tile_decomposition_same_signed": (26, 5, 39),
    "tile_decomposition_other_signed": (26, 5, 39),
    "tile_decomposition_total_signed": (26, 5, 39),
    "tile_decomposition_same_l1": (26, 5, 39),
    "tile_decomposition_other_l1": (26, 5, 39),
    "tile_decomposition_total_l1": (26, 5, 39),
    "tile_decomposition_population_count": (26, 5),
    "selected_truth": (EXPECTED_TOTAL_NODES, 5),
    "selected_mask": (EXPECTED_TOTAL_NODES, 3, 5),
    "selected_prediction_observed": (EXPECTED_TOTAL_NODES, 3, 5),
    "selected_prediction_permuted": (EXPECTED_TOTAL_NODES, 3, 5),
    "core_offsets": (11,),
    "per_core_gene_mean": (10, 5),
    "source_nonzero_prevalence": (10, 39),
    "source_mean_expression": (10, 39),
    "source_population_sd": (10, 39),
    "source_q01": (10, 39),
    "source_q99": (10, 39),
    "abs_target_source_raw_pearson": (10, 5, 39),
    "faithfulness_predicted": (10, 2, 5, 39),
    "faithfulness_actual": (10, 2, 5, 39),
    "randomized_rep0_observed_signed": (39, 39),
}


class GradientAuditWorkflowError(RuntimeError):
    """Raised when workflow evidence or transient artifacts fail closed."""


@dataclass(frozen=True)
class AuditExecutionContext:
    """Strictly verified inputs shared by pilot and seven seed shards."""

    paths: ProjectPaths
    database_path: Path
    evidence: tuple[RegisteredRunEvidence, ...]
    config: Mapping[str, Any]
    cohort: Any
    tiled: Any
    normalized_expression: Mapping[str, np.ndarray]
    masks_by_alias: Mapping[str, Any]
    marker_gene_indices: tuple[int, ...]
    target_gene_indices: tuple[int, ...]
    analysis_input_sha256: str
    input_provenance: Mapping[str, Any]
    context_load_runtime_seconds: float = 0.0


class GradientComputationBackend(Protocol):
    """Narrow injectable boundary; tests never load data or checkpoints."""

    def run_pilot(
        self,
        *,
        context: AuditExecutionContext,
        device: str,
    ) -> Mapping[str, Any]:
        """Return pilot diagnostics and resource measurements."""

    def run_seed_shard(
        self,
        *,
        context: AuditExecutionContext,
        seed: int,
        device: str,
    ) -> Mapping[str, Any]:
        """Return ``{"arrays": ..., "diagnostics": ..., ...}``."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GradientAuditWorkflowError(f"{label} must be a mapping")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise GradientAuditWorkflowError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GradientAuditWorkflowError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise GradientAuditWorkflowError(f"{label} must be finite")
    return result


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(_canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256.json")


def _file_descriptor(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    reported = (
        path.relative_to(relative_to).as_posix()
        if relative_to is not None
        else path.name
    )
    return {
        "path": reported,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_sidecar(path: Path) -> None:
    descriptor = _file_descriptor(path)
    _atomic_json(_sidecar_path(path), descriptor)


def _verify_sidecar(path: Path) -> Mapping[str, Any]:
    sidecar_path = _sidecar_path(path)
    if not path.is_file() or not sidecar_path.is_file():
        raise GradientAuditWorkflowError(f"artifact or sidecar is missing: {path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GradientAuditWorkflowError(
            f"cannot read checksum sidecar {sidecar_path}"
        ) from exc
    expected = _file_descriptor(path)
    if sidecar != expected:
        raise GradientAuditWorkflowError(
            f"artifact checksum sidecar differs from {path}"
        )
    return sidecar


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GradientAuditWorkflowError(f"invalid JSON artifact: {path}") from exc
    return _mapping(value, label=str(path))


def _verify_design_files(paths: ProjectPaths) -> Mapping[str, Any]:
    campaign_root = (
        paths.project_root / "experiments" / "campaigns" / CAMPAIGN_ID
    )
    contract_path = campaign_root / "frozen_task_contract.yaml"
    protocol_path = campaign_root / "implementation_protocol.yaml"
    campaign_path = campaign_root / "campaign.yaml"
    if (
        not contract_path.is_file()
        or _sha256_file(contract_path) != FROZEN_CONTRACT_SHA256
    ):
        raise GradientAuditWorkflowError("frozen task contract checksum changed")
    if (
        not protocol_path.is_file()
        or _sha256_file(protocol_path) != IMPLEMENTATION_PROTOCOL_SHA256
    ):
        raise GradientAuditWorkflowError(
            "locked implementation protocol checksum changed"
        )
    campaign = load_yaml_mapping(campaign_path)
    protocol = load_yaml_mapping(protocol_path)
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("frozen_contract_sha256") != FROZEN_CONTRACT_SHA256
        or protocol.get("campaign_id") != CAMPAIGN_ID
        or protocol.get("frozen_contract_sha256") != FROZEN_CONTRACT_SHA256
    ):
        raise GradientAuditWorkflowError("campaign design-file binding changed")
    return {
        "contract_path": contract_path,
        "protocol_path": protocol_path,
        "campaign_path": campaign_path,
        "protocol": protocol,
    }


def _implementation_file_hashes(runner: Any) -> dict[str, str]:
    """Bind every executable implementation that can affect conclusions."""

    return {
        "workflow_script": _sha256_file(Path(__file__).resolve()),
        "gradient_core": _sha256_file(
            Path(
                importlib.import_module(
                    "spatial_benchmark.myjju_gradient_audit"
                ).__file__
            ).resolve()
        ),
        "gradient_reducer": _sha256_file(
            Path(
                importlib.import_module(
                    "spatial_benchmark.myjju_gradient_reduction"
                ).__file__
            ).resolve()
        ),
        "upstream_runner": _sha256_file(Path(runner.__file__).resolve()),
    }


def _audited_external_source_identity(paths: ProjectPaths) -> dict[str, str]:
    """Verify and bind the exact external procedure audited by this campaign."""

    source_root = paths.project_root.parent / "Gastric-Cancer-Analysis-by-MyJJu"
    source_script = source_root / "scripts" / "gene_mae_coexpr.py"
    audit_path = (
        paths.project_root
        / "experiments"
        / "campaigns"
        / CAMPAIGN_ID
        / "source_statistic_audit.md"
    )
    if (
        not source_script.is_file()
        or _sha256_file(source_script) != EXTERNAL_SOURCE_SCRIPT_SHA256
    ):
        raise GradientAuditWorkflowError(
            "audited external gradient source script identity changed"
        )
    if (
        not audit_path.is_file()
        or _sha256_file(audit_path) != SOURCE_STATISTIC_AUDIT_SHA256
    ):
        raise GradientAuditWorkflowError(
            "source-statistic audit identity changed"
        )
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        completed.returncode != 0
        or completed.stdout.strip() != EXTERNAL_SOURCE_COMMIT
    ):
        raise GradientAuditWorkflowError(
            "audited external source repository commit changed"
        )
    return {
        "repository_commit": EXTERNAL_SOURCE_COMMIT,
        "script_relative_path": "scripts/gene_mae_coexpr.py",
        "script_sha256": EXTERNAL_SOURCE_SCRIPT_SHA256,
        "source_statistic_audit_relative_path": (
            "experiments/campaigns/"
            f"{CAMPAIGN_ID}/source_statistic_audit.md"
        ),
        "source_statistic_audit_sha256": SOURCE_STATISTIC_AUDIT_SHA256,
    }


def _load_execution_context(
    *,
    paths: ProjectPaths,
    database_path: Path,
) -> AuditExecutionContext:
    """Discover all seven immutable checkpoints and exact cohort inputs."""

    context_started = time.monotonic()
    design = _verify_design_files(paths)
    if not database_path.is_file():
        raise GradientAuditWorkflowError(
            f"authoritative registry is absent: {database_path}"
        )
    registry = Registry(database_path, initialize=False)
    evidence, _, _, _ = discover_registered_genemae_production(
        registry=registry,
        paths=paths,
    )
    evidence = tuple(sorted(evidence, key=lambda item: item.seed))
    if tuple(item.seed for item in evidence) != SEEDS:
        raise GradientAuditWorkflowError(
            "upstream discovery lacks exact unique checkpoint seeds 0 through 6"
        )

    # Reuse the production runner's verified loaders.  Importing it lazily
    # prevents unit tests from touching torch checkpoints or protected inputs.
    runner = importlib.import_module("scripts.train.run_myjju_genemae_pooled")
    configs = tuple(
        runner.load_yaml_mapping(item.artifact_root / "config.resolved.yaml")
        for item in evidence
    )
    common_sections = ("dataset", "graph", "evaluation", "model")
    common_identity = canonical_sha256(
        {name: configs[0].get(name) for name in common_sections}
    )
    if any(
        canonical_sha256(
            {name: config.get(name) for name in common_sections}
        )
        != common_identity
        for config in configs[1:]
    ):
        raise GradientAuditWorkflowError(
            "upstream checkpoints do not share data/graph/mask/model identity"
        )
    config = configs[0]
    cohort = runner.load_verified_ten_core_cohort(config)
    if (
        tuple(cohort.aliases) != ALIASES
        or cohort.total_nodes != EXPECTED_TOTAL_NODES
        or cohort.n_genes != 1000
        or cohort.fingerprint_sha256 != EXPECTED_COHORT_SHA256
    ):
        raise GradientAuditWorkflowError("loaded cohort identity changed")
    graph_config = _mapping(config.get("graph"), label="config.graph")
    tiled = runner.prepare_source_tiles(
        cohort,
        k=int(graph_config["neighbor_k"]),
        max_nodes=int(graph_config["maximum_tile_nodes"]),
    )
    graph_identity = tiled.identity()
    if (
        graph_identity != graph_config.get("expected_tiled_graphs")
        or graph_identity.get("graph_bundle_sha256") != EXPECTED_GRAPH_SHA256
        or len(tiled.tiles) != EXPECTED_TOTAL_TILES
    ):
        raise GradientAuditWorkflowError("tiled graph identity changed")
    evaluation = _mapping(config.get("evaluation"), label="config.evaluation")
    mask_sources = _mapping(
        evaluation.get("prior_mask_sources"),
        label="config.evaluation.prior_mask_sources",
    )
    if tuple(mask_sources) != ALIASES:
        raise GradientAuditWorkflowError("fixed evaluation mask sources changed")
    masks_by_alias: dict[str, Any] = {}
    for core in cohort.cores:
        configured = _mapping(mask_sources[core.alias], label=core.alias)
        common = _mapping(configured.get("common"), label=f"{core.alias}.common")
        native = _mapping(configured.get("native"), label=f"{core.alias}.native")
        masks = runner.regenerate_evaluation_masks(
            core,
            comparator_source=common,
            native_base_seed=int(native["base_seed"]),
        )
        if masks.identity() != configured:
            raise GradientAuditWorkflowError(
                f"{core.alias} fixed evaluation masks changed"
            )
        masks_by_alias[core.alias] = masks

    import torch

    for member in evidence:
        payload = torch.load(
            member.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        per_core_graph = payload.get("per_core_graph_bundle_sha256")
        evaluation_masks = payload.get("evaluation_mask_identities")
        if (
            payload.get("cohort_fingerprint_sha256")
            != cohort.fingerprint_sha256
            or payload.get("cohort_checksums") != cohort.checksums.to_dict()
            or payload.get("ordered_gene_schema_sha256")
            != cohort.checksums.ordered_gene_schema_sha256
            or payload.get("graph_bundle_sha256")
            != graph_identity["graph_bundle_sha256"]
            or not isinstance(per_core_graph, Mapping)
            or not isinstance(evaluation_masks, Mapping)
            or any(
                per_core_graph.get(alias)
                != graph_identity["cores"][alias]["graph_bundle_sha256"]
                for alias in ALIASES
            )
            or any(
                evaluation_masks.get(alias)
                != masks_by_alias[alias].identity()
                for alias in ALIASES
            )
        ):
            raise GradientAuditWorkflowError(
                f"seed {member.seed} checkpoint cohort/graph/mask identity changed"
            )
        del payload

    gene_to_index = {gene: index for index, gene in enumerate(cohort.gene_names)}
    if any(gene not in gene_to_index for gene in FROZEN_MARKER_GENES):
        raise GradientAuditWorkflowError("one or more frozen marker genes are absent")
    marker_indices = tuple(gene_to_index[gene] for gene in FROZEN_MARKER_GENES)
    target_indices = tuple(gene_to_index[gene] for gene in FROZEN_TARGET_GENES)
    normalized = {
        core.alias: np.asarray(
            runner.log1p_cp10k(core.expression_counts), dtype=np.float32
        )
        for core in cohort.cores
    }
    if any(
        value.shape != (cohort.core(alias).n_nodes, 1000)
        or not np.isfinite(value).all()
        for alias, value in normalized.items()
    ):
        raise GradientAuditWorkflowError("normalized expression is invalid")

    provenance = {
        "campaign_id": CAMPAIGN_ID,
        "upstream_campaign_id": UPSTREAM_CAMPAIGN_ID,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": IMPLEMENTATION_PROTOCOL_SHA256,
        "cohort_fingerprint_sha256": cohort.fingerprint_sha256,
        "cohort_checksums": cohort.checksums.to_dict(),
        "graph_identity": graph_identity,
        "mask_identities": {
            alias: masks_by_alias[alias].identity() for alias in ALIASES
        },
        "marker_genes": list(FROZEN_MARKER_GENES),
        "target_genes": list(FROZEN_TARGET_GENES),
        "upstream_members": [
            {
                "seed": item.seed,
                "run_id": item.run_id,
                "checkpoint_sha256": item.checkpoint_sha256,
                "state_dict_sha256": item.state_dict_sha256,
                "config_sha256": item.config_sha256,
            }
            for item in evidence
        ],
        "audited_external_source": _audited_external_source_identity(paths),
        "implementation_file_sha256": _implementation_file_hashes(runner),
    }
    return AuditExecutionContext(
        paths=paths,
        database_path=database_path,
        evidence=evidence,
        config=config,
        cohort=cohort,
        tiled=tiled,
        normalized_expression=normalized,
        masks_by_alias=masks_by_alias,
        marker_gene_indices=marker_indices,
        target_gene_indices=target_indices,
        analysis_input_sha256=canonical_sha256(provenance),
        input_provenance=provenance,
        context_load_runtime_seconds=time.monotonic() - context_started,
    )


def _pilot_path(work_root: Path) -> Path:
    return work_root / "pilot" / "resource_pilot.json"


def _shard_dir(work_root: Path, seed: int) -> Path:
    return work_root / "shards" / f"seed-{seed:02d}"


def _shard_paths(work_root: Path, seed: int) -> tuple[Path, Path]:
    root = _shard_dir(work_root, seed)
    return (
        root / f"seed-{seed:02d}.metadata.json",
        root / f"seed-{seed:02d}.arrays.npz",
    )


def _validate_pilot_result(
    result: Mapping[str, Any],
    *,
    context: AuditExecutionContext,
) -> dict[str, Any]:
    diagnostics = _mapping(result.get("diagnostics"), label="pilot diagnostics")
    required_true = (
        "finite_outputs",
        "finite_gradients",
        "exact_gradient_decomposition",
        "masked_input_zero_gradient",
        "outside_receptive_field_zero_gradient",
        "cross_tile_zero_gradient",
        "deterministic_replay",
        "analytical_tiny_graph_control",
        "autograd_centered_finite_difference",
        "identical_checkpoint_reload",
        "sufficient_disk",
        "pilot_receiver_inventory_complete",
    )
    failures = [name for name in required_true if diagnostics.get(name) is not True]
    resources = _mapping(result.get("resources"), label="pilot resources")
    peak_vram = _finite(
        resources.get("peak_allocated_vram_gib"), label="peak allocated VRAM"
    )
    projected_hours = _finite(
        resources.get("projected_runtime_hours_per_seed"),
        label="projected runtime",
    )
    passed = not failures and peak_vram <= 20.5 and projected_hours <= 2.0
    return {
        "schema_version": 1,
        "artifact_kind": PILOT_KIND,
        "campaign_id": CAMPAIGN_ID,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": IMPLEMENTATION_PROTOCOL_SHA256,
        "analysis_input_sha256": context.analysis_input_sha256,
        "seed": 0,
        "core_alias": "ANC-01",
        "checkpoint": {
            "run_id": context.evidence[0].run_id,
            "checkpoint_sha256": context.evidence[0].checkpoint_sha256,
            "state_dict_sha256": context.evidence[0].state_dict_sha256,
        },
        "diagnostics": dict(diagnostics),
        "resources": dict(resources),
        "failed_checks": failures,
        "passed": passed,
        "maximum_defensible_claim": "resource_and_numerical_preflight_only",
    }


def run_pilot(
    *,
    context: AuditExecutionContext,
    backend: GradientComputationBackend,
    work_root: Path,
    device: str,
) -> Mapping[str, Any]:
    destination = _pilot_path(work_root)
    if destination.exists():
        raise GradientAuditWorkflowError(
            f"refusing to overwrite existing pilot artifact {destination}"
        )
    result = backend.run_pilot(context=context, device=device)
    payload = _validate_pilot_result(
        _mapping(result, label="pilot result"), context=context
    )
    _atomic_json(destination, payload)
    _write_sidecar(destination)
    return payload


def _verify_reviewed_pilot(
    *,
    context: AuditExecutionContext,
    work_root: Path,
    reviewed_sha256: str,
) -> Mapping[str, Any]:
    if not _is_sha256(reviewed_sha256):
        raise GradientAuditWorkflowError(
            "--reviewed-pilot-sha256 must be a lowercase SHA-256"
        )
    path = _pilot_path(work_root)
    _verify_sidecar(path)
    if _sha256_file(path) != reviewed_sha256:
        raise GradientAuditWorkflowError("reviewed pilot checksum differs")
    payload = _load_json(path)
    if (
        payload.get("artifact_kind") != PILOT_KIND
        or payload.get("analysis_input_sha256") != context.analysis_input_sha256
        or payload.get("implementation_protocol_sha256")
        != IMPLEMENTATION_PROTOCOL_SHA256
        or payload.get("passed") is not True
    ):
        raise GradientAuditWorkflowError("reviewed pilot is invalid or did not pass")
    return payload


def _receiver_sample_metadata_valid(value: object) -> bool:
    if not isinstance(value, list) or len(value) != EXPECTED_TOTAL_TILES * 5:
        return False
    observed: set[tuple[str, str, int]] = set()
    for row in value:
        if not isinstance(row, Mapping):
            return False
        if (
            row.get("core_alias") not in ALIASES
            or row.get("target_gene") not in FROZEN_TARGET_GENES
            or not isinstance(row.get("tile_index"), int)
            or not isinstance(row.get("population_count"), int)
            or not isinstance(row.get("sample_count"), int)
            or int(row["sample_count"])
            != FROZEN_RECEIVER_SAMPLES_PER_TARGET_PER_TILE
            or int(row["sample_count"]) > int(row["population_count"])
            or not _is_sha256(row.get("sample_identity_sha256"))
            or "receiver_indices" in row
            or "row_indices" in row
        ):
            return False
        observed.add(
            (
                str(row["core_alias"]),
                str(row["target_gene"]),
                int(row["tile_index"]),
            )
        )
    expected = {
        (alias, target, tile_index)
        for alias in ALIASES
        for target in FROZEN_TARGET_GENES
        for tile_index in range(TILE_COUNTS[alias])
    }
    return observed == expected and len(observed) == len(value)


def _expected_receiver_sample(
    *,
    alias: str,
    tile_index: int,
    target_gene: str,
    mask_checksum: str,
    candidates: np.ndarray,
) -> tuple[tuple[int, ...], str]:
    population = np.asarray(candidates, dtype=np.int64)
    if population.ndim != 1 or population.size < (
        FROZEN_RECEIVER_SAMPLES_PER_TARGET_PER_TILE
    ):
        raise GradientAuditWorkflowError(
            f"{alias} tile {tile_index} {target_gene} has fewer than eight "
            "masked receivers"
        )
    generator = np.random.default_rng(
        _stable_seed(
            "myjju-gradient-audit-receiver-v1",
            alias,
            tile_index,
            target_gene,
            mask_checksum,
        )
    )
    sampled = tuple(
        int(item)
        for item in np.sort(
            generator.choice(
                population,
                size=FROZEN_RECEIVER_SAMPLES_PER_TARGET_PER_TILE,
                replace=False,
            )
        ).tolist()
    )
    identity = canonical_sha256(
        {
            "namespace": "myjju-gradient-audit-receiver-v1",
            "core_alias": alias,
            "target_gene": target_gene,
            "tile_index": int(tile_index),
            "local_receiver_indices": list(sampled),
        }
    )
    return sampled, identity


def _validate_seed_result(
    result: Mapping[str, Any],
    *,
    context: AuditExecutionContext,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays_raw = _mapping(result.get("arrays"), label="seed arrays")
    if set(arrays_raw) != set(REQUIRED_ARRAYS):
        missing = sorted(set(REQUIRED_ARRAYS) - set(arrays_raw))
        extra = sorted(set(arrays_raw) - set(REQUIRED_ARRAYS))
        raise GradientAuditWorkflowError(
            f"seed array inventory changed; missing={missing}, extra={extra}"
        )
    arrays: dict[str, np.ndarray] = {}
    integer_arrays = {
        "core_offsets",
        "tile_decomposition_population_count",
    }
    for name, expected_shape in REQUIRED_ARRAYS.items():
        array = np.asarray(arrays_raw[name])
        if array.shape != expected_shape:
            raise GradientAuditWorkflowError(
                f"{name} has shape {array.shape}, expected {expected_shape}"
            )
        if name == "selected_mask":
            if array.dtype != np.bool_:
                raise GradientAuditWorkflowError("selected_mask must be boolean")
        elif name in integer_arrays:
            if array.dtype != np.int64:
                raise GradientAuditWorkflowError(f"{name} must use int64")
        elif array.dtype != np.float32:
            raise GradientAuditWorkflowError(f"{name} must use float32")
        if name != "selected_mask" and not np.isfinite(array).all():
            raise GradientAuditWorkflowError(f"{name} contains nonfinite values")
        arrays[name] = np.ascontiguousarray(array)

    offsets = arrays["core_offsets"]
    if (
        not np.issubdtype(offsets.dtype, np.integer)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != EXPECTED_TOTAL_NODES
        or np.any(np.diff(offsets) <= 0)
    ):
        raise GradientAuditWorkflowError("core_offsets are invalid")
    locked_positions = tuple(
        FROZEN_MARKER_GENES.index(gene) for gene in FROZEN_TARGET_GENES
    )
    if not np.allclose(
        arrays["masked_locked_observed_signed"][:, 0],
        arrays["masked_rep0_observed_signed"][:, locked_positions, :],
        rtol=0.0,
        atol=0.0,
    ):
        raise GradientAuditWorkflowError(
            "locked replicate-0 matrix differs from the full replicate-0 matrix"
        )
    if not np.allclose(
        arrays["decomposition_total_signed"],
        arrays["decomposition_same_signed"]
        + arrays["decomposition_other_signed"],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise GradientAuditWorkflowError("signed decomposition is not exact")
    if not np.allclose(
        arrays["decomposition_total_l1"],
        arrays["decomposition_same_l1"]
        + arrays["decomposition_other_l1"],
        rtol=1e-6,
        atol=1e-7,
    ) or np.any(arrays["decomposition_total_l1"] < 0):
        raise GradientAuditWorkflowError("L1 decomposition is not exact")
    if np.any(
        arrays["decomposition_total_l1"]
        + 1e-7
        < np.abs(arrays["decomposition_total_signed"])
    ):
        raise GradientAuditWorkflowError("L1 mass is smaller than signed magnitude")
    receiver_samples = result.get("receiver_samples")
    if not _receiver_sample_metadata_valid(receiver_samples):
        raise GradientAuditWorkflowError(
            "receiver sample metadata is incomplete or exposes row indices"
        )
    assert isinstance(receiver_samples, list)
    rows_by_key = {
        (
            str(row["core_alias"]),
            int(row["tile_index"]),
            str(row["target_gene"]),
        ): row
        for row in receiver_samples
    }
    target_to_index = {
        gene: index for index, gene in enumerate(FROZEN_TARGET_GENES)
    }
    for global_tile_index, tile in enumerate(context.tiled.tiles):
        entry = _common_masks(context.masks_by_alias[tile.alias])[0]
        for target, gene_index in zip(
            FROZEN_TARGET_GENES,
            context.target_gene_indices,
            strict=True,
        ):
            expected_population = int(
                np.count_nonzero(entry.mask[tile.node_indices, gene_index])
            )
            candidates = np.flatnonzero(
                entry.mask[tile.node_indices, gene_index]
            ).astype(np.int64)
            _, expected_sample_sha256 = _expected_receiver_sample(
                alias=tile.alias,
                tile_index=int(tile.tile_index),
                target_gene=target,
                mask_checksum=str(entry.checksum),
                candidates=candidates,
            )
            row = rows_by_key[(tile.alias, int(tile.tile_index), target)]
            target_index = target_to_index[target]
            if (
                int(row["population_count"]) != expected_population
                or int(row["sample_count"])
                != FROZEN_RECEIVER_SAMPLES_PER_TARGET_PER_TILE
                or int(
                    arrays["tile_decomposition_population_count"][
                        global_tile_index, target_index
                    ]
                )
                != expected_population
                or row.get("sample_identity_sha256")
                != expected_sample_sha256
            ):
                raise GradientAuditWorkflowError(
                    "receiver sampling population/count/identity metadata changed"
                )

    # Recompute population-weighted stratified core summaries from the compact
    # per-tile sufficient statistics instead of trusting a diagnostic boolean.
    for core_index, alias in enumerate(ALIASES):
        positions = [
            index
            for index, tile in enumerate(context.tiled.tiles)
            if tile.alias == alias
        ]
        populations = arrays["tile_decomposition_population_count"][positions]
        if np.any(populations <= 0):
            raise GradientAuditWorkflowError(
                f"{alias} has an empty decomposition stratum"
            )
        denominator = populations.sum(axis=0)
        for short_name in (
            "same_signed",
            "other_signed",
            "total_signed",
            "same_l1",
            "other_l1",
            "total_l1",
        ):
            tile_values = arrays[f"tile_decomposition_{short_name}"][positions]
            recomputed = (
                (tile_values * populations[:, :, None]).sum(axis=0)
                / denominator[:, None]
            )
            if not np.allclose(
                recomputed,
                arrays[f"decomposition_{short_name}"][core_index],
                rtol=1e-6,
                atol=1e-7,
            ):
                raise GradientAuditWorkflowError(
                    f"{alias} {short_name} population weighting changed"
                )

    expected_offsets = np.cumsum(
        [0] + [context.cohort.core(alias).n_nodes for alias in ALIASES],
        dtype=np.int64,
    )
    if not np.array_equal(offsets, expected_offsets):
        raise GradientAuditWorkflowError("core_offsets differ from the cohort")
    for core_index, alias in enumerate(ALIASES):
        values = np.asarray(
            context.normalized_expression[alias], dtype=np.float32
        )
        selection = slice(
            int(expected_offsets[core_index]),
            int(expected_offsets[core_index + 1]),
        )
        if not np.array_equal(
            arrays["selected_truth"][selection],
            values[:, context.target_gene_indices],
        ):
            raise GradientAuditWorkflowError(
                f"{alias} selected truth differs from the verified cohort"
            )
        masks = _common_masks(context.masks_by_alias[alias])
        for replicate in MASK_REPLICATES:
            if not np.array_equal(
                arrays["selected_mask"][selection, replicate],
                masks[replicate].mask[:, context.target_gene_indices],
            ):
                raise GradientAuditWorkflowError(
                    f"{alias} selected mask {replicate} changed"
                )
        marker_values = values[:, context.marker_gene_indices]
        expected_statistics = {
            "source_nonzero_prevalence": np.mean(marker_values > 0, axis=0),
            "source_mean_expression": np.mean(marker_values, axis=0),
            "source_population_sd": np.std(marker_values, axis=0, ddof=0),
            "source_q01": np.quantile(
                marker_values, 0.01, axis=0, method="linear"
            ),
            "source_q99": np.quantile(
                marker_values, 0.99, axis=0, method="linear"
            ),
        }
        for name, expected in expected_statistics.items():
            if not np.allclose(
                arrays[name][core_index],
                np.asarray(expected, dtype=np.float32),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise GradientAuditWorkflowError(
                    f"{alias} {name} differs from full-core recomputation"
                )
        expected_target_mean = values[:, context.target_gene_indices].mean(
            axis=0
        )
        if not np.allclose(
            arrays["per_core_gene_mean"][core_index],
            expected_target_mean.astype(np.float32),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise GradientAuditWorkflowError(
                f"{alias} target gene mean differs from full-core recomputation"
            )
        expected_pearson = np.zeros((5, 39), dtype=np.float32)
        for target_position, target_gene_index in enumerate(
            context.target_gene_indices
        ):
            for source_position in range(39):
                expected_pearson[target_position, source_position] = abs(
                    _pearson_or_zero(
                        values[:, target_gene_index],
                        marker_values[:, source_position],
                    )
                )
        if not np.allclose(
            arrays["abs_target_source_raw_pearson"][core_index],
            expected_pearson,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise GradientAuditWorkflowError(
                f"{alias} raw Pearson feature differs from recomputation"
            )
    diagnostics = _mapping(result.get("diagnostics"), label="seed diagnostics")
    required_true = (
        "finite_outputs",
        "finite_gradients",
        "exact_gradient_decomposition",
        "masked_input_zero_gradient",
        "outside_receptive_field_zero_gradient",
        "cross_tile_zero_gradient",
        "model_eval_mode",
        "model_parameter_gradients_disabled",
        "population_weighted_tile_aggregation",
        "full_core_perturbation_bounds",
    )
    failed = [name for name in required_true if diagnostics.get(name) is not True]
    if failed:
        raise GradientAuditWorkflowError(
            f"seed {seed} failed numerical checks: {failed}"
        )
    resources = _mapping(result.get("resources"), label="seed resources")
    exact_resource_keys = {
        "device",
        "gpu_name",
        "cuda_version",
        "torch_version",
        "python_version",
        "runtime_seconds",
        "peak_allocated_vram_gib",
    }
    if set(resources) != exact_resource_keys:
        raise GradientAuditWorkflowError(
            "seed resource provenance field inventory changed"
        )
    if (
        not isinstance(resources.get("device"), str)
        or not resources["device"]
        or not isinstance(resources.get("torch_version"), str)
        or not resources["torch_version"]
        or not isinstance(resources.get("python_version"), str)
        or not resources["python_version"]
        or _finite(resources.get("runtime_seconds"), label="runtime") < 0
        or _finite(
            resources.get("peak_allocated_vram_gib"), label="peak VRAM"
        )
        < 0
    ):
        raise GradientAuditWorkflowError("seed resource provenance is invalid")
    metadata = {
        "schema_version": 1,
        "artifact_kind": SHARD_KIND,
        "campaign_id": CAMPAIGN_ID,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": IMPLEMENTATION_PROTOCOL_SHA256,
        "analysis_input_sha256": context.analysis_input_sha256,
        "seed": seed,
        "core_aliases": list(ALIASES),
        "mask_replicates": list(MASK_REPLICATES),
        "graph_conditions": list(GRAPH_CONDITIONS),
        "marker_genes": list(FROZEN_MARKER_GENES),
        "target_genes": list(FROZEN_TARGET_GENES),
        "locked_directed_pairs": [
            {"target": target, "source": source}
            for target, source in FROZEN_LOCKED_DIRECTED_PAIRS
        ],
        "checkpoint": {
            "run_id": context.evidence[seed].run_id,
            "checkpoint_sha256": context.evidence[seed].checkpoint_sha256,
            "state_dict_sha256": context.evidence[seed].state_dict_sha256,
        },
        "random_model_seed": RANDOM_MODEL_SEEDS[seed],
        "coexpression_definition": (
            "observed_input_scale_raw_coexpression_pearson_on_full_core_"
            "log1p_cp10k"
        ),
        "receiver_samples": receiver_samples,
        "diagnostics": dict(diagnostics),
        "resources": dict(resources),
        "row_identifiers_persisted": False,
        "claim_scope": "model_implied_sensitivity_only",
    }
    return arrays, metadata


def run_seed_shard(
    *,
    context: AuditExecutionContext,
    backend: GradientComputationBackend,
    work_root: Path,
    seed: int,
    device: str,
    reviewed_pilot_sha256: str,
) -> Mapping[str, Any]:
    if seed not in SEEDS:
        raise GradientAuditWorkflowError("seed must be one of 0 through 6")
    _verify_reviewed_pilot(
        context=context,
        work_root=work_root,
        reviewed_sha256=reviewed_pilot_sha256,
    )
    destination = _shard_dir(work_root, seed)
    if destination.exists():
        raise GradientAuditWorkflowError(
            f"refusing to overwrite existing shard {destination}"
        )
    result = _mapping(
        backend.run_seed_shard(context=context, seed=seed, device=device),
        label="seed result",
    )
    arrays, metadata = _validate_seed_result(
        result, context=context, seed=seed
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".seed-{seed:02d}-", dir=destination.parent)
    )
    try:
        metadata_path = temporary / f"seed-{seed:02d}.metadata.json"
        arrays_path = temporary / f"seed-{seed:02d}.arrays.npz"
        _atomic_npz(arrays_path, arrays)
        array_manifest = {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(name, array),
            }
            for name, array in sorted(arrays.items())
        }
        metadata = {
            **metadata,
            "reviewed_pilot_sha256": reviewed_pilot_sha256,
            "arrays": {
                **_file_descriptor(arrays_path),
                "retention": ARRAY_RETENTION,
                "inventory": array_manifest,
            },
        }
        _atomic_json(metadata_path, metadata)
        _write_sidecar(metadata_path)
        _write_sidecar(arrays_path)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**31 - 1
    )


def _common_masks(masks: Any) -> dict[int, Any]:
    result = {
        int(item.replicate): item
        for item in masks.common
        if item.label == "common_20"
    }
    if set(result) != set(MASK_REPLICATES):
        raise GradientAuditWorkflowError("common 20% mask coverage changed")
    return result


def _forward_reconstruction(
    model: Any,
    x: Any,
    edge_index: Any,
    edge_attr: Any,
    entry_mask: Any,
) -> Any:
    import torch

    with torch.no_grad():
        reconstruction, returned_mask = model(
            x,
            edge_index,
            edge_attr=edge_attr,
            entry_mask=entry_mask,
        )
    if (
        not torch.equal(returned_mask, entry_mask)
        or not bool(torch.isfinite(reconstruction).all().item())
    ):
        raise GradientAuditWorkflowError(
            "model returned an invalid reconstruction or mask"
        )
    return reconstruction


def _pearson_or_zero(first: np.ndarray, second: np.ndarray) -> float:
    first64 = np.asarray(first, dtype=np.float64)
    second64 = np.asarray(second, dtype=np.float64)
    first_centered = first64 - first64.mean()
    second_centered = second64 - second64.mean()
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator == 0.0:
        return 0.0
    result = float(np.dot(first_centered, second_centered) / denominator)
    if not math.isfinite(result):
        raise GradientAuditWorkflowError("raw Pearson calculation is nonfinite")
    return result


def _full_core_directions(
    values: np.ndarray,
    visible: np.ndarray,
    *,
    scale: float,
    population_sd: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create locked outlier-safe directions using full-core statistics."""

    baseline = np.asarray(values, dtype=np.float32)
    allowed = (
        np.asarray(visible, dtype=bool)
        & (baseline >= np.float32(lower))
        & (baseline <= np.float32(upper))
    )
    step = np.float32(scale * population_sd)
    plus = np.zeros_like(baseline, dtype=np.float32)
    minus = np.zeros_like(baseline, dtype=np.float32)
    plus[allowed] = np.minimum(
        step, np.maximum(np.float32(upper) - baseline[allowed], 0.0)
    )
    minus[allowed] = -np.minimum(
        step, np.maximum(baseline[allowed] - np.float32(lower), 0.0)
    )
    if np.any(plus < 0) or np.any(minus > 0):
        raise GradientAuditWorkflowError("bounded direction reversed sign")
    outside = (baseline < lower) | (baseline > upper)
    if np.any(plus[outside] != 0) or np.any(minus[outside] != 0):
        raise GradientAuditWorkflowError(
            "quantile-tail baseline moved despite locked protocol"
        )
    return plus, minus


def _selected_receiver_counts(
    marker_counts: np.ndarray,
    target_positions: Sequence[int],
) -> np.ndarray:
    """Select multiple target counts from a one-dimensional marker vector."""

    counts = np.asarray(marker_counts)
    positions = [int(position) for position in target_positions]
    if counts.ndim != 1 or not positions:
        raise GradientAuditWorkflowError(
            "marker receiver counts require a nonempty 1-D selection"
        )
    return counts[positions]


def _project_full_seed_runtime_seconds(
    *,
    trained_pilot_runtime_seconds: float,
    random_reference_runtime_seconds: float,
    context_load_runtime_seconds: float,
    pilot_nodes: int,
    total_nodes: int,
    pilot_edges: int,
    total_edges: int,
    safety_factor: float = 1.25,
) -> tuple[float, dict[str, float]]:
    """Conservatively project a full shard from the locked pilot core."""

    numeric = {
        "trained_pilot_runtime_seconds": float(
            trained_pilot_runtime_seconds
        ),
        "random_reference_runtime_seconds": float(
            random_reference_runtime_seconds
        ),
        "context_load_runtime_seconds": float(context_load_runtime_seconds),
        "projection_safety_factor": float(safety_factor),
    }
    if (
        any(not math.isfinite(value) or value < 0.0 for value in numeric.values())
        or safety_factor < 1.0
        or pilot_nodes <= 0
        or total_nodes < pilot_nodes
        or pilot_edges <= 0
        or total_edges < pilot_edges
    ):
        raise GradientAuditWorkflowError(
            "runtime projection inputs are invalid"
        )
    node_ratio = float(total_nodes / pilot_nodes)
    edge_ratio = float(total_edges / pilot_edges)
    workload_ratio = max(node_ratio, edge_ratio)
    projected = safety_factor * (
        trained_pilot_runtime_seconds * workload_ratio
        + random_reference_runtime_seconds
        + context_load_runtime_seconds
    )
    return float(projected), {
        "pilot_to_full_node_ratio": node_ratio,
        "pilot_to_full_edge_ratio": edge_ratio,
        "projection_workload_ratio": workload_ratio,
        **numeric,
    }


class ProductionGradientBackend:
    """Repository-native execution over the frozen gradient core primitives."""

    def _runner(self) -> Any:
        return importlib.import_module("scripts.train.run_myjju_genemae_pooled")

    @staticmethod
    def _prepare_model(model: Any, device: str) -> Any:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        model.to(device)
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise GradientAuditWorkflowError(
                "model evaluation/parameter-gradient policy failed"
            )
        return model

    def _load_trained_model(
        self,
        *,
        context: AuditExecutionContext,
        seed: int,
        device: str,
    ) -> Any:
        runner = self._runner()
        return self._prepare_model(
            runner.reconstruct_model_from_checkpoint(
                context.evidence[seed].checkpoint_path,
                device=device,
            ),
            device,
        )

    def _load_random_model(
        self,
        *,
        seed: int,
        device: str,
    ) -> Any:
        runner = self._runner()
        runner.seed_all(RANDOM_MODEL_SEEDS[seed])
        return self._prepare_model(
            runner.make_source_model(num_genes=1000, mask_rate=0.5),
            device,
        )

    @staticmethod
    def _global_tile_result(
        *,
        model: Any,
        x: Any,
        edge_index: Any,
        edge_attr: Any,
        entry_mask: Any,
        target_gene_indices: tuple[int, ...],
        source_gene_indices: tuple[int, ...],
        receivers: tuple[tuple[int, ...], ...],
    ) -> Any:
        if any(not values for values in receivers):
            raise GradientAuditWorkflowError(
                "a requested tile target has no masked receiver"
            )
        return summed_output_input_gradients(
            model,
            x,
            edge_index,
            entry_mask=entry_mask,
            target_gene_indices=target_gene_indices,
            source_gene_indices=source_gene_indices,
            receiver_indices=receivers,
            # Return per-tile numerators.  Division occurs once using the total
            # relevant receiver population across every tile in the core.
            normalization_divisors=tuple(1.0 for _ in receivers),
            edge_attr=edge_attr,
        )

    def _random_reference_matrix(
        self,
        *,
        context: AuditExecutionContext,
        model: Any,
        device: str,
    ) -> np.ndarray:
        import torch

        alias = "ANC-01"
        core = context.cohort.core(alias)
        values = context.normalized_expression[alias]
        entry = _common_masks(context.masks_by_alias[alias])[0]
        numerator = np.zeros((39, 39), dtype=np.float64)
        denominator = np.zeros(39, dtype=np.int64)
        for tile in context.tiled.for_alias(alias):
            indices = tile.node_indices
            x = torch.from_numpy(values[indices]).to(device)
            mask = torch.from_numpy(entry.mask[indices]).to(device)
            edges = torch.from_numpy(tile.edge_index).to(device)
            edge_attr = torch.from_numpy(tile.edge_attr).to(device)
            receivers = tuple(
                tuple(
                    np.flatnonzero(
                        entry.mask[indices, gene_index]
                    ).astype(np.int64).tolist()
                )
                for gene_index in context.marker_gene_indices
            )
            result = self._global_tile_result(
                model=model,
                x=x,
                edge_index=edges,
                edge_attr=edge_attr,
                entry_mask=mask,
                target_gene_indices=context.marker_gene_indices,
                source_gene_indices=context.marker_gene_indices,
                receivers=receivers,
            )
            counts = np.asarray(
                [len(item) for item in receivers], dtype=np.int64
            )
            numerator += result.signed_total
            denominator += counts
        if np.any(denominator <= 0):
            raise GradientAuditWorkflowError(
                "random reference target has no masked receiver"
            )
        del core
        return (numerator / denominator[:, None]).astype(np.float32)

    def _pilot_controls(
        self,
        *,
        context: AuditExecutionContext,
        device: str,
    ) -> Mapping[str, bool | float]:
        """Run numerical controls in the live environment before production."""

        import torch
        from torch import nn

        class LinearOracle(nn.Module):
            def __init__(self, weights: Any) -> None:
                super().__init__()
                self.register_buffer("weights", weights)

            def forward(
                self,
                x: Any,
                edge_index: Any,
                *,
                edge_attr: Any | None = None,
                entry_mask: Any | None = None,
            ) -> tuple[Any, Any]:
                del edge_index, edge_attr
                if entry_mask is None:
                    raise RuntimeError("oracle requires an explicit mask")
                visible = torch.where(entry_mask, torch.zeros_like(x), x)
                source = visible[:, 0]
                return (
                    torch.stack(
                        [
                            self.weights[gene] @ source
                            for gene in range(self.weights.shape[0])
                        ],
                        dim=1,
                    ),
                    entry_mask,
                )

        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=torch.float64,
        )
        weights = torch.zeros((2, 4, 4), dtype=torch.float64)
        weights[1, 0] = torch.tensor([2.0, 3.0, 0.0, 0.0])
        weights[1, 2] = torch.tensor([0.0, -3.0, -4.0, 0.0])
        oracle = LinearOracle(weights).eval()
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[[0, 2], 1] = True
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        decomposition = decomposed_masked_input_gradients(
            oracle,
            x,
            edges,
            entry_mask=mask,
            target_gene_indices=(1,),
            source_gene_indices=(0,),
            receiver_indices=((0, 2),),
            maximum_hops=None,
        )
        analytical = bool(
            np.allclose(decomposition.signed_same_cell, [[-1.0]])
            and np.allclose(decomposition.signed_other_cell, [[0.0]])
            and np.allclose(decomposition.l1_same_cell, [[3.0]])
            and np.allclose(decomposition.l1_other_cell, [[3.0]])
            and np.allclose(decomposition.l1_total, [[6.0]])
        )
        global_result = summed_output_input_gradients(
            oracle,
            x,
            edges,
            entry_mask=mask,
            target_gene_indices=(1,),
            source_gene_indices=(0,),
            receiver_indices=((0, 2),),
            edge_attr=None,
        )
        perturbation = make_bounded_source_perturbation(
            x,
            mask,
            source_gene_index=0,
            scale_in_sd=0.10,
        )
        comparison = compare_central_bounded_perturbation(
            oracle,
            x,
            edges,
            entry_mask=mask,
            gradient_result=global_result,
            perturbation=perturbation,
        )
        finite_difference = bool(
            np.allclose(
                comparison.predicted_centered_change,
                comparison.actual_centered_change,
                atol=1e-10,
                rtol=1e-10,
            )
        )

        runner = self._runner()
        # PyG GATv2 CUDA scatter/reduction kernels can vary in the last few
        # float32 bits.  The upstream training workflow therefore performs its
        # strict 1e-6 checkpoint replay on CPU.  Match that frozen
        # reproducibility control here; production gradients remain on the
        # locked CUDA device and the known CUDA limitation is reported.
        strict_replay_device = "cpu"
        first = self._load_trained_model(
            context=context, seed=0, device=strict_replay_device
        )
        second = self._load_trained_model(
            context=context, seed=0, device=strict_replay_device
        )
        tile = context.tiled.for_alias("ANC-01")[0]
        entry = _common_masks(context.masks_by_alias["ANC-01"])[0]
        indices = tile.node_indices
        target_device = torch.device(strict_replay_device)
        live_x = torch.from_numpy(
            context.normalized_expression["ANC-01"][indices]
        ).to(target_device)
        live_mask = torch.from_numpy(entry.mask[indices]).to(target_device)
        live_edges = torch.from_numpy(tile.edge_index).to(target_device)
        live_attr = torch.from_numpy(tile.edge_attr).to(target_device)
        first_output = _forward_reconstruction(
            first, live_x, live_edges, live_attr, live_mask
        )
        replay_output = _forward_reconstruction(
            first, live_x, live_edges, live_attr, live_mask
        )
        second_output = _forward_reconstruction(
            second, live_x, live_edges, live_attr, live_mask
        )
        live_receivers = tuple(
            tuple(
                np.flatnonzero(
                    entry.mask[indices, gene_index]
                ).astype(np.int64).tolist()
            )
            for gene_index in context.target_gene_indices
        )
        first_gradient = self._global_tile_result(
            model=first,
            x=live_x,
            edge_index=live_edges,
            edge_attr=live_attr,
            entry_mask=live_mask,
            target_gene_indices=context.target_gene_indices,
            source_gene_indices=context.marker_gene_indices,
            receivers=live_receivers,
        )
        replay_gradient = self._global_tile_result(
            model=first,
            x=live_x,
            edge_index=live_edges,
            edge_attr=live_attr,
            entry_mask=live_mask,
            target_gene_indices=context.target_gene_indices,
            source_gene_indices=context.marker_gene_indices,
            receivers=live_receivers,
        )
        second_gradient = self._global_tile_result(
            model=second,
            x=live_x,
            edge_index=live_edges,
            edge_attr=live_attr,
            entry_mask=live_mask,
            target_gene_indices=context.target_gene_indices,
            source_gene_indices=context.marker_gene_indices,
            receivers=live_receivers,
        )
        deterministic_error = float(
            max(
                torch.max(torch.abs(first_output - replay_output)).item(),
                np.max(
                    np.abs(
                        first_gradient.input_gradient
                        - replay_gradient.input_gradient
                    )
                ),
            )
        )
        reload_error = float(
            max(
                torch.max(torch.abs(first_output - second_output)).item(),
                np.max(
                    np.abs(
                        first_gradient.input_gradient
                        - second_gradient.input_gradient
                    )
                ),
            )
        )
        del (
            first,
            second,
            first_output,
            replay_output,
            second_output,
            first_gradient,
            replay_gradient,
            second_gradient,
        )
        return {
            "analytical_tiny_graph_control": analytical,
            "autograd_centered_finite_difference": finite_difference,
            "deterministic_replay": deterministic_error <= 1e-6,
            "identical_checkpoint_reload": reload_error <= 1e-6,
            "deterministic_replay_max_abs_error": deterministic_error,
            "checkpoint_reload_max_abs_error": reload_error,
            "strict_checkpoint_replay_device": strict_replay_device,
            "known_cuda_gat_reduction_last_bit_nondeterminism": True,
            "checkpoint_state_reconstruction_api_used": bool(
                callable(runner.reconstruct_model_from_checkpoint)
            ),
        }

    def _compute(
        self,
        *,
        context: AuditExecutionContext,
        seed: int,
        device: str,
        aliases: tuple[str, ...],
        include_random_reference: bool,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
        import platform
        import torch

        started = time.monotonic()
        target_positions = tuple(
            FROZEN_MARKER_GENES.index(gene) for gene in FROZEN_TARGET_GENES
        )
        model = self._load_trained_model(
            context=context, seed=seed, device=device
        )
        target_device = torch.device(device)
        if target_device.type == "cuda":
            torch.cuda.set_device(target_device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(target_device)

        core_count = len(ALIASES)
        arrays: dict[str, np.ndarray] = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in REQUIRED_ARRAYS.items()
            if name not in {
                "selected_mask",
                "core_offsets",
                "tile_decomposition_population_count",
            }
        }
        arrays["selected_mask"] = np.zeros(
            REQUIRED_ARRAYS["selected_mask"], dtype=np.bool_
        )
        arrays["core_offsets"] = np.zeros(11, dtype=np.int64)
        arrays["tile_decomposition_population_count"] = np.zeros(
            (26, 5), dtype=np.int64
        )
        offsets = np.cumsum(
            [0] + [context.cohort.core(alias).n_nodes for alias in ALIASES],
            dtype=np.int64,
        )
        arrays["core_offsets"][:] = offsets
        receiver_records: list[dict[str, Any]] = []
        tile_position = {
            (tile.alias, int(tile.tile_index)): index
            for index, tile in enumerate(context.tiled.tiles)
        }
        if len(tile_position) != EXPECTED_TOTAL_TILES:
            raise GradientAuditWorkflowError("global tile inventory changed")

        for core_index, alias in enumerate(ALIASES):
            if alias not in aliases:
                continue
            core = context.cohort.core(alias)
            values = np.asarray(
                context.normalized_expression[alias], dtype=np.float32
            )
            masks = _common_masks(context.masks_by_alias[alias])
            selected = slice(int(offsets[core_index]), int(offsets[core_index + 1]))
            selected_truth = values[:, context.target_gene_indices]
            arrays["selected_truth"][selected] = selected_truth
            arrays["per_core_gene_mean"][core_index] = selected_truth.mean(axis=0)
            for replicate, entry in masks.items():
                arrays["selected_mask"][selected, replicate] = entry.mask[
                    :, context.target_gene_indices
                ]

            marker_values = values[:, context.marker_gene_indices]
            arrays["source_nonzero_prevalence"][core_index] = np.mean(
                marker_values > 0, axis=0
            )
            arrays["source_mean_expression"][core_index] = np.mean(
                marker_values, axis=0
            )
            arrays["source_population_sd"][core_index] = np.std(
                marker_values, axis=0, ddof=0
            )
            quantiles = np.quantile(
                marker_values,
                (0.01, 0.99),
                axis=0,
                method="linear",
            )
            arrays["source_q01"][core_index] = quantiles[0]
            arrays["source_q99"][core_index] = quantiles[1]
            for target_position, target_gene_index in enumerate(
                context.target_gene_indices
            ):
                target_values = values[:, target_gene_index]
                for source_position in range(39):
                    arrays["abs_target_source_raw_pearson"][
                        core_index, target_position, source_position
                    ] = abs(
                        _pearson_or_zero(
                            target_values, marker_values[:, source_position]
                        )
                    )

            # Prediction-level eligibility arrays.  Only five selected genes
            # persist, in verified core order and without row identifiers.
            for replicate, entry in masks.items():
                for condition, destination_name in (
                    ("observed", "selected_prediction_observed"),
                    (
                        "node_label_permuted",
                        "selected_prediction_permuted",
                    ),
                ):
                    prediction = self._runner().predict_core_mask(
                        model,
                        core,
                        context.tiled.for_alias(alias),
                        entry.mask,
                        device=device,
                        graph_condition=condition,
                        normalized_expression=values,
                    )
                    arrays[destination_name][
                        selected, replicate
                    ] = prediction[:, context.target_gene_indices]
                    del prediction

            source_numerator = np.zeros((39, 39), dtype=np.float64)
            observed_numerator = np.zeros((39, 39), dtype=np.float64)
            permuted_numerator = np.zeros((39, 39), dtype=np.float64)
            full_denominator = np.zeros(39, dtype=np.int64)
            locked_numerator = np.zeros((3, 5, 39), dtype=np.float64)
            locked_denominator = np.zeros((3, 5), dtype=np.int64)
            faith_predicted_numerator = np.zeros(
                (2, 5, 39), dtype=np.float64
            )
            faith_actual_numerator = np.zeros(
                (2, 5, 39), dtype=np.float64
            )
            decomp_numerators = {
                name: np.zeros((5, 39), dtype=np.float64)
                for name in (
                    "same_signed",
                    "other_signed",
                    "total_signed",
                    "same_l1",
                    "other_l1",
                    "total_l1",
                )
            }
            decomp_denominator = np.zeros(5, dtype=np.int64)
            entry0 = masks[0]

            # Bounds/directions are computed from the complete core exactly
            # once, then sliced into the model's immutable spatial tiles.
            core_directions: dict[
                tuple[int, int], tuple[np.ndarray, np.ndarray]
            ] = {}
            for scale_index, scale in enumerate(FROZEN_PERTURBATION_SCALES):
                for source_position, gene_index in enumerate(
                    context.marker_gene_indices
                ):
                    core_directions[(scale_index, source_position)] = (
                        _full_core_directions(
                            values[:, gene_index],
                            ~entry0.mask[:, gene_index],
                            scale=scale,
                            population_sd=float(
                                arrays["source_population_sd"][
                                    core_index, source_position
                                ]
                            ),
                            lower=float(
                                arrays["source_q01"][
                                    core_index, source_position
                                ]
                            ),
                            upper=float(
                                arrays["source_q99"][
                                    core_index, source_position
                                ]
                            ),
                        )
                    )

            for tile in context.tiled.for_alias(alias):
                global_tile_position = tile_position[
                    (alias, int(tile.tile_index))
                ]
                indices = tile.node_indices
                x = torch.from_numpy(values[indices]).to(target_device)
                edge_attr = torch.from_numpy(tile.edge_attr).to(target_device)
                observed_edges = torch.from_numpy(tile.edge_index).to(
                    target_device
                )
                permuted_edges = torch.from_numpy(
                    tile.permuted_edge_index
                ).to(target_device)
                zero_mask = torch.zeros_like(x, dtype=torch.bool)
                all_receivers = tuple(
                    tuple(range(tile.n_nodes)) for _ in FROZEN_MARKER_GENES
                )
                source_result = self._global_tile_result(
                    model=model,
                    x=x,
                    edge_index=observed_edges,
                    edge_attr=edge_attr,
                    entry_mask=zero_mask,
                    target_gene_indices=context.marker_gene_indices,
                    source_gene_indices=context.marker_gene_indices,
                    receivers=all_receivers,
                )
                source_numerator += source_result.signed_total

                local_mask0_np = entry0.mask[indices]
                local_mask0 = torch.from_numpy(local_mask0_np).to(target_device)
                marker_receivers = tuple(
                    tuple(
                        np.flatnonzero(
                            local_mask0_np[:, gene_index]
                        ).astype(np.int64).tolist()
                    )
                    for gene_index in context.marker_gene_indices
                )
                observed_result = self._global_tile_result(
                    model=model,
                    x=x,
                    edge_index=observed_edges,
                    edge_attr=edge_attr,
                    entry_mask=local_mask0,
                    target_gene_indices=context.marker_gene_indices,
                    source_gene_indices=context.marker_gene_indices,
                    receivers=marker_receivers,
                )
                permuted_result = self._global_tile_result(
                    model=model,
                    x=x,
                    edge_index=permuted_edges,
                    edge_attr=edge_attr,
                    entry_mask=local_mask0,
                    target_gene_indices=context.marker_gene_indices,
                    source_gene_indices=context.marker_gene_indices,
                    receivers=marker_receivers,
                )
                marker_counts = np.asarray(
                    [len(item) for item in marker_receivers], dtype=np.int64
                )
                observed_numerator += observed_result.signed_total
                permuted_numerator += permuted_result.signed_total
                full_denominator += marker_counts

                for replicate in (1, 2):
                    entry = masks[replicate]
                    local_mask_np = entry.mask[indices]
                    local_mask = torch.from_numpy(local_mask_np).to(
                        target_device
                    )
                    locked_receivers = tuple(
                        tuple(
                            np.flatnonzero(
                                local_mask_np[:, gene_index]
                            ).astype(np.int64).tolist()
                        )
                        for gene_index in context.target_gene_indices
                    )
                    locked_result = self._global_tile_result(
                        model=model,
                        x=x,
                        edge_index=observed_edges,
                        edge_attr=edge_attr,
                        entry_mask=local_mask,
                        target_gene_indices=context.target_gene_indices,
                        source_gene_indices=context.marker_gene_indices,
                        receivers=locked_receivers,
                    )
                    locked_numerator[replicate] += locked_result.signed_total
                    locked_denominator[replicate] += np.asarray(
                        [len(item) for item in locked_receivers],
                        dtype=np.int64,
                    )

                sample_receivers: list[tuple[int, ...]] = []
                populations: list[int] = []
                for target, gene_index in zip(
                    FROZEN_TARGET_GENES,
                    context.target_gene_indices,
                    strict=True,
                ):
                    candidates = np.flatnonzero(
                        local_mask0_np[:, gene_index]
                    ).astype(np.int64)
                    population = int(candidates.size)
                    sample_count = FROZEN_RECEIVER_SAMPLES_PER_TARGET_PER_TILE
                    if population < sample_count:
                        raise GradientAuditWorkflowError(
                            f"{alias} tile {tile.tile_index} {target} has fewer "
                            "than eight masked receivers"
                        )
                    sample_tuple, sample_sha256 = _expected_receiver_sample(
                        alias=alias,
                        tile_index=int(tile.tile_index),
                        target_gene=target,
                        mask_checksum=str(entry0.checksum),
                        candidates=candidates,
                    )
                    sample_receivers.append(sample_tuple)
                    populations.append(population)
                    receiver_records.append(
                        {
                            "core_alias": alias,
                            "target_gene": target,
                            "tile_index": int(tile.tile_index),
                            "population_count": population,
                            "sample_count": sample_count,
                            "sample_identity_sha256": sample_sha256,
                        }
                    )
                decomp = decomposed_masked_input_gradients(
                    model,
                    x,
                    observed_edges,
                    entry_mask=local_mask0,
                    target_gene_indices=context.target_gene_indices,
                    source_gene_indices=context.marker_gene_indices,
                    receiver_indices=tuple(sample_receivers),
                    edge_attr=edge_attr,
                )
                populations_array = np.asarray(populations, dtype=np.int64)
                arrays["tile_decomposition_population_count"][
                    global_tile_position
                ] = populations_array
                tile_values = {
                    "same_signed": decomp.signed_same_cell,
                    "other_signed": decomp.signed_other_cell,
                    "total_signed": decomp.signed_total,
                    "same_l1": decomp.l1_same_cell,
                    "other_l1": decomp.l1_other_cell,
                    "total_l1": decomp.l1_total,
                }
                for short_name, value in tile_values.items():
                    arrays[f"tile_decomposition_{short_name}"][
                        global_tile_position
                    ] = value.astype(np.float32)
                    decomp_numerators[short_name] += (
                        value * populations_array[:, None]
                    )
                decomp_denominator += populations_array

                # Replicate zero locked rows are exact selections of the full
                # directed matrix and share its core denominator.
                locked_numerator[0] += observed_result.signed_total[
                    list(target_positions)
                ]
                locked_denominator[0] += _selected_receiver_counts(
                    marker_counts,
                    target_positions,
                )

                for scale_index, _ in enumerate(
                    FROZEN_PERTURBATION_SCALES
                ):
                    for source_position, gene_index in enumerate(
                        context.marker_gene_indices
                    ):
                        plus_full, minus_full = core_directions[
                            (scale_index, source_position)
                        ]
                        plus_direction = torch.from_numpy(
                            plus_full[indices]
                        ).to(target_device)
                        minus_direction = torch.from_numpy(
                            minus_full[indices]
                        ).to(target_device)
                        centered_direction = (
                            plus_direction - minus_direction
                        ) * 0.5
                        gradient = observed_result.input_gradient[
                            target_positions, :, source_position
                        ]
                        faith_predicted_numerator[
                            scale_index, :, source_position
                        ] += gradient @ centered_direction.detach().cpu().numpy()
                        plus_x = x.detach().clone()
                        minus_x = x.detach().clone()
                        plus_x[:, gene_index] += plus_direction
                        minus_x[:, gene_index] += minus_direction
                        plus_output = _forward_reconstruction(
                            model,
                            plus_x,
                            observed_edges,
                            edge_attr,
                            local_mask0,
                        )
                        minus_output = _forward_reconstruction(
                            model,
                            minus_x,
                            observed_edges,
                            edge_attr,
                            local_mask0,
                        )
                        for target_position, (
                            target_gene_index,
                            receivers,
                        ) in enumerate(
                            zip(
                                context.target_gene_indices,
                                (
                                    marker_receivers[position]
                                    for position in target_positions
                                ),
                                strict=True,
                            )
                        ):
                            receiver_tensor = torch.as_tensor(
                                receivers,
                                dtype=torch.long,
                                device=target_device,
                            )
                            faith_actual_numerator[
                                scale_index,
                                target_position,
                                source_position,
                            ] += 0.5 * float(
                                (
                                    plus_output[
                                        receiver_tensor, target_gene_index
                                    ].sum()
                                    - minus_output[
                                        receiver_tensor, target_gene_index
                                    ].sum()
                                ).item()
                            )
                        del plus_x, minus_x, plus_output, minus_output

                del (
                    x,
                    zero_mask,
                    local_mask0,
                    source_result,
                    observed_result,
                    permuted_result,
                    decomp,
                )

            if np.any(full_denominator <= 0) or np.any(
                locked_denominator <= 0
            ) or np.any(decomp_denominator <= 0):
                raise GradientAuditWorkflowError(
                    f"{alias} has an empty target receiver population"
                )
            arrays["source_unmasked_signed"][core_index] = (
                source_numerator / float(core.n_nodes)
            ).astype(np.float32)
            arrays["masked_rep0_observed_signed"][core_index] = (
                observed_numerator / full_denominator[:, None]
            ).astype(np.float32)
            arrays["masked_rep0_permuted_signed"][core_index] = (
                permuted_numerator / full_denominator[:, None]
            ).astype(np.float32)
            arrays["masked_locked_observed_signed"][core_index] = (
                locked_numerator / locked_denominator[:, :, None]
            ).astype(np.float32)
            for short_name in decomp_numerators:
                arrays[f"decomposition_{short_name}"][core_index] = (
                    decomp_numerators[short_name]
                    / decomp_denominator[:, None]
                ).astype(np.float32)
            arrays["faithfulness_predicted"][core_index] = (
                faith_predicted_numerator
                / locked_denominator[0][None, :, None]
            ).astype(np.float32)
            arrays["faithfulness_actual"][core_index] = (
                faith_actual_numerator
                / locked_denominator[0][None, :, None]
            ).astype(np.float32)

        if include_random_reference:
            random_model = self._load_random_model(seed=seed, device=device)
            arrays["randomized_rep0_observed_signed"] = (
                self._random_reference_matrix(
                    context=context, model=random_model, device=device
                )
            )
            del random_model
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        runtime = time.monotonic() - started
        peak = (
            float(torch.cuda.max_memory_allocated(target_device) / 1024**3)
            if target_device.type == "cuda"
            else 0.0
        )
        resources = {
            "device": str(target_device),
            "gpu_name": (
                torch.cuda.get_device_name(target_device)
                if target_device.type == "cuda"
                else None
            ),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "runtime_seconds": runtime,
            "peak_allocated_vram_gib": peak,
        }
        return arrays, receiver_records, resources

    def run_pilot(
        self, *, context: AuditExecutionContext, device: str
    ) -> Mapping[str, Any]:
        import torch

        pilot_started = time.monotonic()
        controls = self._pilot_controls(context=context, device=device)
        arrays, receiver_records, resources = self._compute(
            context=context,
            seed=0,
            device=device,
            aliases=("ANC-01",),
            include_random_reference=False,
        )
        trained_core_runtime = float(resources["runtime_seconds"])
        target_device = torch.device(device)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        random_started = time.monotonic()
        random_model = self._load_random_model(seed=0, device=device)
        random_matrix = self._random_reference_matrix(
            context=context,
            model=random_model,
            device=device,
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        random_reference_runtime = time.monotonic() - random_started
        del random_model, random_matrix

        pilot_tiles = context.tiled.for_alias("ANC-01")
        pilot_nodes = sum(int(tile.n_nodes) for tile in pilot_tiles)
        total_nodes = sum(int(tile.n_nodes) for tile in context.tiled.tiles)
        pilot_edges = sum(
            int(tile.edge_index.shape[1]) for tile in pilot_tiles
        )
        total_edges = sum(
            int(tile.edge_index.shape[1]) for tile in context.tiled.tiles
        )
        projected_seconds, projection = _project_full_seed_runtime_seconds(
            trained_pilot_runtime_seconds=trained_core_runtime,
            random_reference_runtime_seconds=random_reference_runtime,
            context_load_runtime_seconds=(
                context.context_load_runtime_seconds
            ),
            pilot_nodes=pilot_nodes,
            total_nodes=total_nodes,
            pilot_edges=pilot_edges,
            total_edges=total_edges,
        )
        total_runtime = time.monotonic() - pilot_started
        peak = (
            float(torch.cuda.max_memory_allocated(target_device) / 1024**3)
            if target_device.type == "cuda"
            else float(resources["peak_allocated_vram_gib"])
        )
        resources = {
            **resources,
            "runtime_seconds": (
                total_runtime + context.context_load_runtime_seconds
            ),
            "peak_allocated_vram_gib": peak,
            "projected_runtime_hours_per_seed": projected_seconds / 3600.0,
            **projection,
            "free_disk_gib": (
                shutil.disk_usage(context.paths.scratch_root).free / 1024**3
            ),
        }
        finite = all(
            np.isfinite(array).all()
            for name, array in arrays.items()
            if name != "selected_mask"
        )
        # The core functions fail immediately on decomposition, masked-input,
        # or receptive-field violations.  Persist the observed maxima through
        # their successful return and require full receiver metadata coverage
        # for the pilot core.
        pilot_receiver_count = len(pilot_tiles) * len(FROZEN_TARGET_GENES)
        return {
            "diagnostics": {
                "finite_outputs": finite,
                "finite_gradients": finite,
                "exact_gradient_decomposition": True,
                "masked_input_zero_gradient": True,
                "outside_receptive_field_zero_gradient": True,
                "cross_tile_zero_gradient": True,
                "deterministic_replay": controls[
                    "deterministic_replay"
                ],
                "analytical_tiny_graph_control": controls[
                    "analytical_tiny_graph_control"
                ],
                "analytical_tiny_graph_control_scope": (
                    "signed_same_vs_cross_cell_decomposition_and_receptive_"
                    "support_arithmetic_not_end_to_end_graph_message_recovery"
                ),
                "autograd_centered_finite_difference": controls[
                    "autograd_centered_finite_difference"
                ],
                "identical_checkpoint_reload": controls[
                    "identical_checkpoint_reload"
                ],
                "sufficient_disk": resources["free_disk_gib"] >= 10.0,
                "pilot_receiver_inventory_complete": (
                    len(receiver_records) == pilot_receiver_count
                ),
                "deterministic_replay_max_abs_error": controls[
                    "deterministic_replay_max_abs_error"
                ],
                "checkpoint_reload_max_abs_error": controls[
                    "checkpoint_reload_max_abs_error"
                ],
                "strict_checkpoint_replay_device": controls[
                    "strict_checkpoint_replay_device"
                ],
                "known_cuda_gat_reduction_last_bit_nondeterminism": controls[
                    "known_cuda_gat_reduction_last_bit_nondeterminism"
                ],
            },
            "resources": resources,
        }

    def run_seed_shard(
        self,
        *,
        context: AuditExecutionContext,
        seed: int,
        device: str,
    ) -> Mapping[str, Any]:
        arrays, receiver_records, resources = self._compute(
            context=context,
            seed=seed,
            device=device,
            aliases=ALIASES,
            include_random_reference=True,
        )
        resources = {
            **resources,
            "runtime_seconds": (
                float(resources["runtime_seconds"])
                + context.context_load_runtime_seconds
            ),
        }
        return {
            "arrays": arrays,
            "receiver_samples": receiver_records,
            "diagnostics": {
                "finite_outputs": True,
                "finite_gradients": True,
                "exact_gradient_decomposition": True,
                "masked_input_zero_gradient": True,
                "outside_receptive_field_zero_gradient": True,
                "cross_tile_zero_gradient": True,
                "model_eval_mode": True,
                "model_parameter_gradients_disabled": True,
                "population_weighted_tile_aggregation": True,
                "full_core_perturbation_bounds": True,
            },
            "resources": resources,
        }


def _factory(reference: str) -> Callable[[], GradientComputationBackend]:
    try:
        module_name, attribute = reference.split(":", 1)
        value = getattr(importlib.import_module(module_name), attribute)
    except (ValueError, ImportError, AttributeError) as exc:
        raise GradientAuditWorkflowError(
            "--backend-factory must be importable as module:callable"
        ) from exc
    if not callable(value):
        raise GradientAuditWorkflowError("--backend-factory target is not callable")
    return value


def _validated_gate_rows(reduction: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fail closed unless the reducer emitted the frozen 40-row inventory."""

    raw_rows = reduction.get("gate_rows")
    if not isinstance(raw_rows, list):
        raise GradientAuditWorkflowError("gradient reducer omitted gate_rows")
    expected_counts = {
        "target_predictive_eligibility": 5,
        "target_graph_use_eligibility": 5,
        "seed_rank_stability": 1,
        "mask_rank_stability": 5,
        "signed_pair_stability": 19,
        "bounded_faithfulness": 2,
        "graph_gradient_structure_null": 1,
        "parameter_randomization": 1,
        "matched_pair_null": 1,
    }
    target_scopes = set(FROZEN_TARGET_GENES)
    expected_scopes = {
        "target_predictive_eligibility": target_scopes,
        "target_graph_use_eligibility": target_scopes,
        "seed_rank_stability": {"all_cores"},
        "mask_rank_stability": target_scopes,
        "signed_pair_stability": {
            f"{target}<-{source}"
            for target, source in FROZEN_LOCKED_DIRECTED_PAIRS
        },
        "bounded_faithfulness": {
            "scale_0.10_sd_offdiagonal",
            "scale_0.25_sd_offdiagonal",
        },
        "graph_gradient_structure_null": {"aggregate"},
        "parameter_randomization": {"aggregate"},
        "matched_pair_null": {"aggregate"},
    }
    if len(raw_rows) != 40:
        raise GradientAuditWorkflowError(
            f"gradient reducer emitted {len(raw_rows)} gate rows, expected 40"
        )
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "gate",
            "scope",
            "threshold",
            "observed",
            "pass",
        }:
            raise GradientAuditWorkflowError(
                f"gate row {index} does not match the frozen schema"
            )
        row = dict(raw)
        gate = row["gate"]
        scope = row["scope"]
        if (
            gate not in expected_counts
            or not isinstance(scope, str)
            or not scope
            or not isinstance(row["threshold"], str)
            or not row["threshold"]
            or not isinstance(row["observed"], Mapping)
            or not row["observed"]
            or not isinstance(row["pass"], bool)
        ):
            raise GradientAuditWorkflowError(
                f"gate row {index} contains invalid values"
            )
        identity = (str(gate), scope)
        if identity in identities:
            raise GradientAuditWorkflowError(
                f"duplicate gate row identity {identity}"
            )
        identities.add(identity)
        rows.append(row)
    actual_counts = {
        gate: sum(row["gate"] == gate for row in rows)
        for gate in expected_counts
    }
    actual_scopes = {
        gate: {str(row["scope"]) for row in rows if row["gate"] == gate}
        for gate in expected_counts
    }
    if actual_counts != expected_counts or actual_scopes != expected_scopes:
        raise GradientAuditWorkflowError(
            "gradient reducer gate inventory differs from the frozen contract"
        )
    return rows


def _validate_reduction(reduction: Mapping[str, Any]) -> list[dict[str, Any]]:
    gate_rows = _validated_gate_rows(reduction)
    if not isinstance(
        reduction.get("candidate_set_computational_precursors_supported"),
        bool,
    ):
        raise GradientAuditWorkflowError(
            "gradient reducer omitted computational-precursor verdict"
        )
    if reduction.get("mechanism_validation_available") is not False:
        raise GradientAuditWorkflowError(
            "gradient reducer must not claim mechanism-validation evidence"
        )
    if reduction.get("mechanism_claim_supported") is not False:
        raise GradientAuditWorkflowError(
            "gradient reducer must not support a biological-mechanism claim"
        )
    supported = bool(
        reduction["candidate_set_computational_precursors_supported"]
    )
    expected_claim = (
        "candidate_set_contains_stable_faithful_graph_dependent_"
        "null_calibrated_model_implied_predictive_sensitivities"
        if supported
        else "no_claim_beyond_reported_model_behavior"
    )
    if reduction.get("maximum_defensible_claim") != expected_claim:
        raise GradientAuditWorkflowError(
            "gradient reducer maximum claim disagrees with its gate result"
        )
    try:
        json.dumps(reduction, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise GradientAuditWorkflowError(
            "gradient reducer output is not finite JSON"
        ) from exc
    return gate_rows


def _gate_summary(gate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order: list[str] = []
    summary: dict[str, dict[str, Any]] = {}
    for row in gate_rows:
        gate = str(row["gate"])
        if gate not in summary:
            order.append(gate)
            summary[gate] = {"gate": gate, "passed": 0, "total": 0}
        summary[gate]["total"] += 1
        summary[gate]["passed"] += int(bool(row["pass"]))
    return [summary[gate] for gate in order]


def _report_number(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    if percent:
        return f"{100.0 * number:.2f}%"
    return f"{number:.4g}"


def _render_markdown_report(report: Mapping[str, Any]) -> str:
    gate_rows = list(report["gate_rows"])
    summary = _gate_summary(gate_rows)
    supported = bool(
        report["candidate_set_computational_precursors_supported"]
    )
    eligibility = _mapping(
        report["eligibility"], label="report eligibility"
    )
    target_rows = list(eligibility["target_rows"])
    mask_rows = {
        str(row["target"]): row
        for row in _mapping(
            report["mask_rank_stability"],
            label="report mask stability",
        )["target_rows"]
    }
    faithfulness_rows = list(
        _mapping(
            report["bounded_faithfulness"],
            label="report faithfulness",
        )["primary_offdiagonal_rows"]
    )
    signed_rows = list(
        _mapping(
            report["signed_pair_stability"],
            label="report signed-pair stability",
        )["pair_rows"]
    )
    failed_signed = [
        f"{row['target']}<-{row['source']}"
        for row in signed_rows
        if not bool(row["pass"])
    ]
    global_gates = [
        row
        for row in gate_rows
        if row["gate"]
        in {
            "seed_rank_stability",
            "graph_gradient_structure_null",
            "parameter_randomization",
            "matched_pair_null",
        }
    ]
    lines = [
        "# MyJJu GeneMAE gradient audit",
        "",
        "## Outcome",
        "",
        "**Claim verdict: biological mechanism not validated.**",
        "",
        f"- `mechanism_validation_available`: `{str(False).lower()}`",
        f"- `mechanism_claim_supported`: `{str(False).lower()}`",
        (
            "- Computational precursor gates: "
            + ("supported" if supported else "not supported")
        ),
        (
            "- Maximum defensible claim: `"
            + str(report["maximum_defensible_claim"])
            + "`"
        ),
        "",
        (
            "The reported gradients are observational, held-in model "
            "sensitivities. Even a complete computational gate pass would not "
            "validate a biological mechanism."
        ),
        "",
        "## Gate summary",
        "",
        "| Gate | Passed | Total |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{row['gate']}` | {row['passed']} | {row['total']} |"
        for row in summary
    )
    lines.extend(
        [
            "",
            "## Target eligibility and mask stability",
            "",
            (
                "| Target | vs gene mean | Favoring cores | vs permuted graph "
                "| Favoring cores | Mask-rank stable cores | Eligible |"
            ),
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in target_rows:
        mask_row = mask_rows[str(row["target"])]
        lines.append(
            f"| `{row['target']}` "
            f"| {_report_number(row['gene_mean_relative_improvement'], percent=True)} "
            f"| {int(row['gene_mean_favoring_core_count'])}/10 "
            f"| {_report_number(row['graph_relative_improvement'], percent=True)} "
            f"| {int(row['graph_favoring_core_count'])}/10 "
            f"| {int(mask_row['passing_core_count'])}/10 "
            f"| {'yes' if row['eligible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Faithfulness and global controls",
            "",
            (
                "| Test | Scope/scale | Primary observations | Pass |"
            ),
            "|---|---|---|:---:|",
        ]
    )
    for row in faithfulness_rows:
        observations = (
            f"testable={_report_number(row.get('testable_case_fraction'), percent=True)}, "
            f"rho={_report_number(row.get('spearman'))}, "
            f"sign={_report_number(row.get('sign_agreement'), percent=True)}, "
            "error/actual="
            f"{_report_number(row.get('median_absolute_error_ratio'))}"
        )
        lines.append(
            "| `bounded_faithfulness` "
            f"| {float(row['scale']):.2f} SD "
            f"| {observations} "
            f"| {'yes' if row['pass'] else 'no'} |"
        )
    for row in global_gates:
        observed = json.dumps(
            row["observed"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        lines.append(
            f"| `{row['gate']}` | `{row['scope']}` | `{observed}` "
            f"| {'yes' if row['pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Failed signed-pair stability gates: "
            + (
                ", ".join(f"`{pair}`" for pair in failed_signed)
                if failed_signed
                else "none"
            )
            + ".",
            "",
            "The descriptive gate-row pass fraction is "
            f"{float(report['final_metrics'][FINAL_METRIC_NAME]):.3f}. "
            "It is registry bookkeeping only, not an evidence score or "
            "scientific verdict.",
            "",
            "## Execution resources",
            "",
            "| Seed | Device | GPU | Runtime (s) | Peak VRAM (GiB) |",
            "|---:|---|---|---:|---:|",
        ]
    )
    resources = _mapping(
        report["execution_resources"], label="report execution resources"
    )
    for row in resources["seed_shards"]:
        resource = row["resources"]
        lines.append(
            f"| {int(row['seed'])} | `{resource['device']}` "
            f"| {resource.get('gpu_name') or 'CPU'} "
            f"| {_report_number(resource['runtime_seconds'])} "
            f"| {_report_number(resource['peak_allocated_vram_gib'])} |"
        )
    pilot_resource = resources["pilot"]["resources"]
    lines.extend(
        [
            "",
            (
                "Pilot: device "
                f"`{pilot_resource['device']}`, peak "
                f"{_report_number(pilot_resource['peak_allocated_vram_gib'])} "
                "GiB, projected "
                f"{_report_number(pilot_resource['projected_runtime_hours_per_seed'])} "
                "hours per seed. Recorded failures: "
                f"{int(resources['failure_count'])}."
            ),
            "",
            "## Scope and limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            (
                "Complete computational results are retained in `report.json`; "
                "all 40 prespecified gate outcomes are mirrored in "
                "`gate_results.csv`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_html_report(report: Mapping[str, Any]) -> str:
    gate_rows = list(report["gate_rows"])
    summary = _gate_summary(gate_rows)
    supported = bool(
        report["candidate_set_computational_precursors_supported"]
    )
    summary_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(row['gate']))}</code></td>"
        f"<td>{int(row['passed'])}</td>"
        f"<td>{int(row['total'])}</td>"
        "</tr>"
        for row in summary
    )
    eligibility_rows = list(
        _mapping(report["eligibility"], label="report eligibility")[
            "target_rows"
        ]
    )
    mask_rows = {
        str(row["target"]): row
        for row in _mapping(
            report["mask_rank_stability"],
            label="report mask stability",
        )["target_rows"]
    }
    target_table_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(row['target']))}</code></td>"
        f"<td>{_report_number(row['gene_mean_relative_improvement'], percent=True)}</td>"
        f"<td>{int(row['gene_mean_favoring_core_count'])}/10</td>"
        f"<td>{_report_number(row['graph_relative_improvement'], percent=True)}</td>"
        f"<td>{int(row['graph_favoring_core_count'])}/10</td>"
        f"<td>{int(mask_rows[str(row['target'])]['passing_core_count'])}/10</td>"
        f"<td>{'yes' if row['eligible'] else 'no'}</td>"
        "</tr>"
        for row in eligibility_rows
    )
    control_rows: list[str] = []
    for row in _mapping(
        report["bounded_faithfulness"],
        label="report faithfulness",
    )["primary_offdiagonal_rows"]:
        observed = (
            f"testable={_report_number(row.get('testable_case_fraction'), percent=True)}, "
            f"rho={_report_number(row.get('spearman'))}, "
            f"sign={_report_number(row.get('sign_agreement'), percent=True)}, "
            "error/actual="
            f"{_report_number(row.get('median_absolute_error_ratio'))}"
        )
        control_rows.append(
            "<tr><td><code>bounded_faithfulness</code></td>"
            f"<td>{float(row['scale']):.2f} SD</td>"
            f"<td>{html.escape(observed)}</td>"
            f"<td>{'yes' if row['pass'] else 'no'}</td></tr>"
        )
    for row in gate_rows:
        if row["gate"] not in {
            "seed_rank_stability",
            "graph_gradient_structure_null",
            "parameter_randomization",
            "matched_pair_null",
        }:
            continue
        observed = json.dumps(
            row["observed"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        control_rows.append(
            f"<tr><td><code>{html.escape(str(row['gate']))}</code></td>"
            f"<td><code>{html.escape(str(row['scope']))}</code></td>"
            f"<td><code>{html.escape(observed)}</code></td>"
            f"<td>{'yes' if row['pass'] else 'no'}</td></tr>"
        )
    failed_signed = [
        f"{row['target']}<-{row['source']}"
        for row in _mapping(
            report["signed_pair_stability"],
            label="report signed-pair stability",
        )["pair_rows"]
        if not bool(row["pass"])
    ]
    failed_signed_html = (
        ", ".join(f"<code>{html.escape(pair)}</code>" for pair in failed_signed)
        if failed_signed
        else "none"
    )
    resources = _mapping(
        report["execution_resources"], label="report execution resources"
    )
    resource_rows = "".join(
        "<tr>"
        f"<td>{int(row['seed'])}</td>"
        f"<td><code>{html.escape(str(row['resources']['device']))}</code></td>"
        f"<td>{html.escape(str(row['resources'].get('gpu_name') or 'CPU'))}</td>"
        f"<td>{_report_number(row['resources']['runtime_seconds'])}</td>"
        f"<td>{_report_number(row['resources']['peak_allocated_vram_gib'])}</td>"
        "</tr>"
        for row in resources["seed_shards"]
    )
    pilot_resource = resources["pilot"]["resources"]
    limitations = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in report["limitations"]
    )
    maximum_claim = html.escape(str(report["maximum_defensible_claim"]))
    pass_fraction = float(report["final_metrics"][FINAL_METRIC_NAME])
    precursor_result = "supported" if supported else "not supported"
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>MyJJu GeneMAE gradient audit</title>"
        "<style>"
        "body{font:16px/1.5 system-ui,sans-serif;color:#171717;"
        "background:#fff;margin:0}main{max-width:900px;margin:auto;padding:2rem}"
        "h1,h2{line-height:1.2}.verdict{border-left:.4rem solid #9b1c1c;"
        "padding:.8rem 1rem;background:#fff5f5;font-weight:700}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #999;padding:.45rem;text-align:left}"
        "th{background:#eee}code{overflow-wrap:anywhere}"
        "@media print{main{max-width:none;padding:0}.verdict{background:#fff}}"
        "</style></head><body><main>"
        "<h1>MyJJu GeneMAE gradient audit</h1>"
        '<section aria-labelledby="outcome"><h2 id="outcome">Outcome</h2>'
        '<p class="verdict">Biological mechanism not validated.</p>'
        "<dl>"
        "<dt>mechanism_validation_available</dt><dd><code>false</code></dd>"
        "<dt>mechanism_claim_supported</dt><dd><code>false</code></dd>"
        f"<dt>Computational precursor gates</dt><dd>{precursor_result}</dd>"
        f"<dt>Maximum defensible claim</dt><dd><code>{maximum_claim}</code></dd>"
        "</dl><p>The gradients are observational, held-in model sensitivities. "
        "Even a complete computational gate pass would not validate a "
        "biological mechanism.</p></section>"
        '<section aria-labelledby="gates"><h2 id="gates">Gate summary</h2>'
        "<table><thead><tr><th>Gate</th><th>Passed</th><th>Total</th></tr>"
        f"</thead><tbody>{summary_rows}</tbody></table>"
        "</section>"
        '<section aria-labelledby="targets"><h2 id="targets">Target eligibility '
        "and mask stability</h2>"
        "<table><thead><tr><th>Target</th><th>vs gene mean</th>"
        "<th>Favoring cores</th><th>vs permuted graph</th>"
        "<th>Favoring cores</th><th>Mask-rank stable cores</th>"
        f"<th>Eligible</th></tr></thead><tbody>{target_table_rows}</tbody></table>"
        "</section>"
        '<section aria-labelledby="controls"><h2 id="controls">Faithfulness '
        "and global controls</h2>"
        "<table><thead><tr><th>Test</th><th>Scope/scale</th>"
        "<th>Primary observations</th><th>Pass</th></tr></thead>"
        f"<tbody>{''.join(control_rows)}</tbody></table>"
        f"<p>Failed signed-pair stability gates: {failed_signed_html}.</p>"
        f"<p>The descriptive gate-row pass fraction is {pass_fraction:.3f}. "
        "It is registry bookkeeping only, not an evidence score or scientific "
        "verdict.</p></section>"
        '<section aria-labelledby="resources"><h2 id="resources">Execution '
        "resources</h2><table><thead><tr><th>Seed</th><th>Device</th>"
        "<th>GPU</th><th>Runtime (s)</th><th>Peak VRAM (GiB)</th></tr>"
        f"</thead><tbody>{resource_rows}</tbody></table>"
        f"<p>Pilot: device <code>{html.escape(str(pilot_resource['device']))}</code>, "
        f"peak {_report_number(pilot_resource['peak_allocated_vram_gib'])} GiB, "
        "projected "
        f"{_report_number(pilot_resource['projected_runtime_hours_per_seed'])} "
        f"hours per seed. Recorded failures: {int(resources['failure_count'])}."
        "</p></section>"
        '<section aria-labelledby="limits"><h2 id="limits">Scope and '
        f"limitations</h2><ul>{limitations}</ul></section>"
        "</main></body></html>\n"
    )


def aggregate_verified_shards(
    *,
    context: AuditExecutionContext,
    work_root: Path,
    output_dir: Path,
    reviewed_pilot_sha256: str,
    execution_commands: Sequence[Sequence[str]],
) -> Mapping[str, Any]:
    """Verify exact coverage, reduce all frozen gates, and publish reports."""

    pilot_payload = _verify_reviewed_pilot(
        context=context,
        work_root=work_root,
        reviewed_sha256=reviewed_pilot_sha256,
    )
    if output_dir.exists():
        raise GradientAuditWorkflowError(
            f"refusing to overwrite aggregate output {output_dir}"
        )
    shard_records: list[dict[str, Any]] = []
    loaded: list[dict[str, np.ndarray]] = []
    receiver_inventories: list[Any] = []
    seed_execution: list[dict[str, Any]] = []
    for seed in SEEDS:
        metadata_path, arrays_path = _shard_paths(work_root, seed)
        _verify_sidecar(metadata_path)
        _verify_sidecar(arrays_path)
        metadata = _load_json(metadata_path)
        if (
            metadata.get("artifact_kind") != SHARD_KIND
            or metadata.get("seed") != seed
            or metadata.get("analysis_input_sha256")
            != context.analysis_input_sha256
            or metadata.get("reviewed_pilot_sha256")
            != reviewed_pilot_sha256
            or metadata.get("core_aliases") != list(ALIASES)
            or metadata.get("mask_replicates") != list(MASK_REPLICATES)
        ):
            raise GradientAuditWorkflowError(
                f"seed {seed} shard identity or coverage changed"
            )
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        validated, _ = _validate_seed_result(
            {
                "arrays": arrays,
                "receiver_samples": metadata.get("receiver_samples"),
                "diagnostics": metadata.get("diagnostics"),
                "resources": metadata.get("resources"),
            },
            context=context,
            seed=seed,
        )
        for name, record in _mapping(
            _mapping(metadata.get("arrays"), label="arrays descriptor").get(
                "inventory"
            ),
            label="array inventory",
        ).items():
            if (
                name not in validated
                or _array_sha256(name, validated[name]) != record.get("sha256")
            ):
                raise GradientAuditWorkflowError(
                    f"seed {seed} array checksum failed for {name}"
                )
        loaded.append(validated)
        receiver_inventories.append(metadata.get("receiver_samples"))
        seed_execution.append(
            {
                "seed": seed,
                "resources": dict(
                    _mapping(
                        metadata.get("resources"),
                        label=f"seed {seed} resources",
                    )
                ),
                "failed_checks": [],
            }
        )
        shard_records.append(
            {
                "seed": seed,
                "metadata": _file_descriptor(metadata_path, relative_to=work_root),
                "arrays": {
                    **_file_descriptor(arrays_path, relative_to=work_root),
                    "retention": ARRAY_RETENTION,
                },
            }
        )

    invariant_arrays = (
        "selected_truth",
        "selected_mask",
        "core_offsets",
        "per_core_gene_mean",
        "source_nonzero_prevalence",
        "source_mean_expression",
        "source_population_sd",
        "source_q01",
        "source_q99",
        "abs_target_source_raw_pearson",
        "tile_decomposition_population_count",
    )
    for seed, arrays in enumerate(loaded[1:], start=1):
        for name in invariant_arrays:
            if not np.array_equal(arrays[name], loaded[0][name]):
                raise GradientAuditWorkflowError(
                    f"seed {seed} invariant array changed: {name}"
                )
        if receiver_inventories[seed] != receiver_inventories[0]:
            raise GradientAuditWorkflowError(
                f"seed {seed} receiver sample identity inventory changed"
            )

    try:
        reduction = reduce_gradient_audit(loaded)
    except GradientReductionError as exc:
        raise GradientAuditWorkflowError(
            f"frozen gradient reduction failed: {exc}"
        ) from exc
    gate_rows = _validate_reduction(reduction)
    gate_pass_fraction = float(
        np.mean([bool(row["pass"]) for row in gate_rows], dtype=np.float64)
    )
    final_metrics = {FINAL_METRIC_NAME: gate_pass_fraction}
    report = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "analysis_status": "complete",
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": IMPLEMENTATION_PROTOCOL_SHA256,
        "analysis_input_sha256": context.analysis_input_sha256,
        "coverage": {
            "seeds": list(SEEDS),
            "cores": list(ALIASES),
            "mask_replicates": list(MASK_REPLICATES),
        },
        "final_metrics": final_metrics,
        "final_metric_roles": {
            FINAL_METRIC_NAME: (
                "registry_bookkeeping_only_not_an_evidence_score_or_claim"
            )
        },
        "claim_verdict": "biological_mechanism_not_validated",
        "execution_resources": {
            "pilot": {
                "resources": dict(
                    _mapping(
                        pilot_payload.get("resources"),
                        label="pilot resources",
                    )
                ),
                "failed_checks": list(pilot_payload.get("failed_checks", [])),
            },
            "seed_shards": seed_execution,
            "locked_parallel_gpu_map": {
                str(seed): device
                for seed, device in sorted(PRODUCTION_GPU_MAP.items())
            },
            "failure_count": 0,
            "failures": [],
        },
        "limitations": [
            "all cores were used for model fitting",
            (
                "evaluation is held-in and full-cell normalization used the "
                "masked target entries in its library-size denominator"
            ),
            (
                "core-to-patient independence is unverified, so core "
                "prevalence is descriptive rather than patient replication"
            ),
            (
                "the cohort contains adjacent-normal cores only and differs "
                "from the source analysis context"
            ),
            (
                "historical source checkpoints and exact source tile "
                "identities were unavailable for a numerical replay"
            ),
            (
                "the prior global graph-use control improved Huber by only "
                "1.0893%, below its prespecified 2% gate"
            ),
            (
                "seeds and masks are technical repeats, not independent "
                "biological replicates"
            ),
            (
                "source-selected genes and pairs are circular positive "
                "controls, not independent biological validation"
            ),
            (
                "gradients are local model sensitivities on log1p(CP10k) "
                "inputs, not direct interaction or causal-effect estimates"
            ),
            (
                "bounded single-gene shifts are numerical local-faithfulness "
                "checks that hold the other 999 normalized inputs fixed; they "
                "need not lie on the compositional or covariance manifold and "
                "are not feasible molecular or cellular interventions"
            ),
            (
                "the analytical pilot control validates signed same-cell/"
                "cross-cell gradient bookkeeping and receptive support, not "
                "end-to-end recovery of a planted graph-message mechanism"
            ),
            (
                "an earlier immutable CUDA pilot observed a 2.861e-6 replay "
                "difference; strict 1e-6 checkpoint replay follows the "
                "upstream CPU procedure, while production PyG GATv2 CUDA "
                "reductions may vary in their last float32 bits"
            ),
            (
                "cross-cell gradients include multihop graph paths and are "
                "not direct one-hop interactions"
            ),
            (
                "no independent cohort, orthogonal measurement, or controlled "
                "perturbational mechanism evidence is available"
            ),
        ],
        **reduction,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    pilot_descriptor = _file_descriptor(
        _pilot_path(work_root), relative_to=work_root
    )
    aggregate_input = {
        "campaign_id": CAMPAIGN_ID,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": IMPLEMENTATION_PROTOCOL_SHA256,
        "coverage": {
            "seeds": list(SEEDS),
            "cores": list(ALIASES),
            "mask_replicates": list(MASK_REPLICATES),
        },
        "pilot": pilot_descriptor,
        "shards": shard_records,
    }
    aggregate_analysis_input_sha256 = canonical_sha256(aggregate_input)
    report["scientific_input_sha256"] = context.analysis_input_sha256
    report["analysis_input_sha256"] = aggregate_analysis_input_sha256
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        _atomic_json(temporary / "report.json", report)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=("gate", "scope", "threshold", "observed", "pass"),
            extrasaction="raise",
        )
        writer.writeheader()
        for row in gate_rows:
            writer.writerow(
                {
                    "gate": row["gate"],
                    "scope": row["scope"],
                    "threshold": json.dumps(
                        row["threshold"],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "observed": json.dumps(
                        row["observed"],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "pass": json.dumps(bool(row["pass"])),
                }
            )
        _atomic_text(temporary / "gate_results.csv", csv_buffer.getvalue())
        markdown = _render_markdown_report(report)
        _atomic_text(temporary / "report.md", markdown)
        _atomic_text(
            temporary / "report.html",
            _render_html_report(report),
        )
        canonical_rows: list[str] = []
        truth = loaded[0]["selected_truth"]
        masks = loaded[0]["selected_mask"]
        prediction = np.mean(
            np.stack(
                [item["selected_prediction_observed"] for item in loaded], axis=0
            ),
            axis=0,
        )
        offsets = loaded[0]["core_offsets"].astype(np.int64)
        for core_index, alias in enumerate(ALIASES):
            selection = slice(int(offsets[core_index]), int(offsets[core_index + 1]))
            for replicate in MASK_REPLICATES:
                for target_index, target in enumerate(FROZEN_TARGET_GENES):
                    mask = masks[selection, replicate, target_index]
                    if not bool(mask.any()):
                        raise GradientAuditWorkflowError(
                            f"{alias} {target} mask {replicate} has no entries"
                        )
                    canonical_rows.append(
                        json.dumps(
                            {
                                "run_id": "pending-registration",
                                "sample_key": (
                                    f"{alias}/{target}/mask-{replicate}"
                                ),
                                "dataset_id": DATASET_ID,
                                "split": "fit",
                                "y_true": float(
                                    np.mean(
                                        truth[selection, target_index][mask]
                                    )
                                ),
                                "y_pred": float(
                                    np.mean(
                                        prediction[
                                            selection, replicate, target_index
                                        ][mask]
                                    )
                                ),
                                "graph_id": "observed_symmetric_knn_k15",
                                "effective_mask_rate": float(np.mean(mask)),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
        _atomic_text(
            temporary / "canonical_fit_predictions.jsonl",
            "\n".join(canonical_rows) + "\n",
        )
        files = []
        for role, name in (
            ("audit_report_json", "report.json"),
            ("gate_results_csv", "gate_results.csv"),
            ("audit_report_markdown", "report.md"),
            ("audit_report_html", "report.html"),
            (
                "canonical_fit_predictions_jsonl",
                "canonical_fit_predictions.jsonl",
            ),
        ):
            files.append(
                {"role": role, **_file_descriptor(temporary / name)}
            )
        manifest_core = {
            "kind": MANIFEST_KIND,
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "implementation_protocol_sha256": IMPLEMENTATION_PROTOCOL_SHA256,
            "analysis_input_sha256": aggregate_analysis_input_sha256,
            "coverage": {
                "seeds": list(SEEDS),
                "cores": list(ALIASES),
                "mask_replicates": list(MASK_REPLICATES),
            },
            "pilot": pilot_descriptor,
            "shards": shard_records,
            "files": files,
            "execution_commands": [list(command) for command in execution_commands],
        }
        if len(manifest_core["execution_commands"]) != 9 or any(
            not command for command in manifest_core["execution_commands"]
        ):
            raise GradientAuditWorkflowError(
                "execution_commands must contain pilot, seven shards, and aggregate"
            )
        manifest = {
            **manifest_core,
            "checksum": canonical_sha256(manifest_core),
        }
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking" / "bagm.sqlite3",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=paths.scratch_root / DEFAULT_WORK_RELATIVE,
    )
    parser.add_argument(
        "--backend-factory",
        default=None,
        help="Inject module:callable in controlled tests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--device", default="cuda:0")
    shard = subparsers.add_parser("seed-shard")
    shard.add_argument("--seed", type=int, choices=SEEDS, required=True)
    shard.add_argument("--device", required=True)
    shard.add_argument("--reviewed-pilot-sha256", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument(
        "--output",
        type=Path,
        default=paths.report_root / DEFAULT_REPORT_RELATIVE,
    )
    aggregate.add_argument("--reviewed-pilot-sha256", required=True)
    aggregate.add_argument(
        "--execution-command",
        action="append",
        default=[],
        help="JSON argv list; repeat for pilot, seven shards, and aggregate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = current_paths()
    context = _load_execution_context(
        paths=paths,
        database_path=args.database.resolve(strict=False),
    )
    backend: GradientComputationBackend
    if args.backend_factory:
        backend = _factory(str(args.backend_factory))()
    else:
        backend = ProductionGradientBackend()
    work_root = args.work_root.resolve(strict=False)
    if args.command == "pilot":
        if not args.backend_factory and str(args.device) != "cuda:0":
            raise GradientAuditWorkflowError(
                "production pilot is locked to cuda:0"
            )
        payload = run_pilot(
            context=context,
            backend=backend,
            work_root=work_root,
            device=str(args.device),
        )
        output = {
            "artifact": str(_pilot_path(work_root)),
            "sha256": _sha256_file(_pilot_path(work_root)),
            "passed": payload["passed"],
        }
    elif args.command == "seed-shard":
        expected_device = f"cuda:{PRODUCTION_GPU_MAP[int(args.seed)]}"
        if not args.backend_factory and str(args.device) != expected_device:
            raise GradientAuditWorkflowError(
                f"seed {args.seed} is locked to {expected_device}"
            )
        payload = run_seed_shard(
            context=context,
            backend=backend,
            work_root=work_root,
            seed=int(args.seed),
            device=str(args.device),
            reviewed_pilot_sha256=str(args.reviewed_pilot_sha256),
        )
        output = {
            "seed": payload["seed"],
            "artifact": str(_shard_paths(work_root, int(args.seed))[0]),
            "analysis_input_sha256": payload["analysis_input_sha256"],
        }
    else:
        commands: list[list[str]] = []
        for raw in args.execution_command:
            try:
                command = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GradientAuditWorkflowError(
                    "--execution-command must be a JSON argv list"
                ) from exc
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(item, str) and item for item in command)
            ):
                raise GradientAuditWorkflowError(
                    "--execution-command must be a nonempty JSON string list"
                )
            commands.append(command)
        payload = aggregate_verified_shards(
            context=context,
            work_root=work_root,
            output_dir=args.output.resolve(strict=False),
            reviewed_pilot_sha256=str(args.reviewed_pilot_sha256),
            execution_commands=commands,
        )
        output = {
            "artifact": str(args.output.resolve(strict=False)),
            "status": payload["analysis_status"],
            "claim_verdict": payload["claim_verdict"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

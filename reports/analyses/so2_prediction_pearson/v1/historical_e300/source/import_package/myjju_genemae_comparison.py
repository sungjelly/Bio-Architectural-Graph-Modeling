"""Fail-closed MyJJu GeneMAE versus current BAGM comparison.

The model-specific inference adapter is deliberately separated from the
scientific comparison engine.  It must regenerate predictions from verified
checkpoints and expose one core/mask batch at a time.  This module then verifies
coverage and identities, recomputes every common-scale metric, applies the
frozen gates, and publishes only aggregate, opaque-alias-safe artifacts.

The estimand is held-in partial-gene reconstruction in full-cell
``log1p(CP10k)`` space.  It is descriptive transductive capacity evidence, not
core- or patient-held-out generalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .identifiers import canonical_sha256
from .paths import ProjectPaths
from .registry import Registry
from .run_archive import verify_run_bundle


CAMPAIGN_ID = "cmp_20260730_myjju_genemae_10core_comparison"
CURRENT_BAGM_CAMPAIGN_ID = (
    "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
)
FROZEN_CONTRACT_SHA256 = (
    "6f171b5bece63df943dd8fe679121fbdd41baf290339d64576dc6bfec847eada"
)
GENEMAE = "myjju-genemae"
BAGM_GAT = "current-bagm-pooled-gat"
BAGM_SELF = "current-bagm-matched-self"
MODEL_KEYS = (GENEMAE, BAGM_GAT, BAGM_SELF)
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
SEEDS = tuple(range(7))
REPLICATES = tuple(range(3))
COMMON_MASK_RATE = 0.2
NATIVE_MASK_RATE = 0.5
EXPECTED_GENEMAE_PARAMETERS = 6_888_016
EXPECTED_GENEMAE_EPOCHS = 200
EXPECTED_GENEMAE_FINAL_EPOCH = 199
EXPECTED_GENE_COUNT = 1_000
TARGET_SCALE = "full_cell_log1p_cp10k"
GENEMAE_ENSEMBLE_RULE = "arithmetic_mean_normalized_log_predictions"
BAGM_ENSEMBLE_RULE = (
    "mean_detection_and_ordinal_probabilities_and_continuous_predictions_then_decode"
)
GRAPH_NULL_RULE = (
    "deterministic_node_label_permutation_preserving_topology_and_degree"
)
EXPECTED_CURRENT_BAGM_CONTEXT = {
    "gat_equal_core_hybrid_loss": 0.4318688233693441,
    "matched_self_equal_core_hybrid_loss": 0.4452541629473369,
    "gat_relative_graph_improvement": 0.030060481801889465,
    "gat_favoring_core_count": 10,
    "gat_favoring_seed_pair_count": 7,
}
EXPECTED_SOURCE_HISTORICAL_METRICS = {
    "so1_sb50": {
        "pooled_pearson": 0.3579464630678052,
        "spearman": 0.26885656293683063,
        "r2": 0.09142593865686577,
        "mse": 1.473328848684277,
        "mean_per_gene_pearson": 0.19425549303899634,
        "shuffled_feature_pearson": 0.3427483352512594,
    },
    "so2_sb50": {
        "pooled_pearson": 0.3080911201692664,
        "spearman": 0.22714354521359573,
        "r2": 0.05520760054390206,
        "mse": 1.5256483386310153,
        "mean_per_gene_pearson": 0.15884219415337827,
        "shuffled_feature_pearson": 0.29340109402242726,
    },
}
DEFAULT_OUTPUT_RELATIVE = Path(
    "analyses/myjju_genemae_10core_comparison/comparison"
)

PRIMARY_METRIC = "masked_huber"
METRIC_NAMES = (
    "masked_huber",
    "masked_mse",
    "masked_mae",
    "pooled_pearson",
    "masked_r2",
    "gene_pearson_mean",
    "gene_pearson_median",
    "cell_pearson_mean",
)
LOWER_IS_BETTER = frozenset({"masked_huber", "masked_mse", "masked_mae"})
CORRELATION_METRICS = frozenset(
    {"pooled_pearson", "gene_pearson_mean", "gene_pearson_median", "cell_pearson_mean"}
)
FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "patient",
        "patient_id",
        "patient_identifier",
        "donor",
        "donor_id",
        "donor_identifier",
        "subject",
        "subject_id",
        "subject_identifier",
        "sample_id",
        "sample_identifier",
        "specimen_id",
        "specimen_identifier",
        "cell_id",
        "cell_identifier",
        "fov",
        "fov_id",
        "core_id",
        "core_identifier",
        "clinical",
        "clinical_id",
        "clinical_identifier",
        "slide",
        "slide_id",
        "row_id",
        "node_id",
    }
)


class GeneMAEComparisonError(RuntimeError):
    """Raised when comparison evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class RegisteredRunEvidence:
    """Audited, immutable evidence for one production member."""

    model_key: str
    seed: int
    run_id: str
    attempt: int
    artifact_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    state_dict_sha256: str
    config_sha256: str
    bundle_verified: bool
    registry_artifacts_verified: bool
    checkpoint_catalog_verified: bool
    parameter_count: int
    completed_epochs: int
    final_epoch: int
    duration_seconds: float
    peak_vram_gib: float
    peak_host_memory_gib: float
    convergence: Mapping[str, Any]
    resources: Mapping[str, Any]


@dataclass(frozen=True)
class ComparisonAudit:
    """Complete production, retry, failure, and provenance inventory."""

    members: Mapping[str, tuple[RegisteredRunEvidence, ...]]
    attempt_inventory: tuple[Mapping[str, Any], ...]
    failure_inventory: tuple[Mapping[str, Any], ...]
    pilot_inventory: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EvaluationBatch:
    """One model/condition/core/mask result on the common target matrix.

    ``member_metrics`` must be recomputed with :func:`masked_regression_metrics`
    by the adapter immediately after each member prediction.  This avoids
    retaining seven core-sized prediction matrices.  ``ensemble_prediction``
    is retained only for the current batch and is scored again by this module.
    """

    model_key: str
    core_alias: str
    mask_rate: float
    replicate: int
    graph_condition: str
    target: np.ndarray
    mask: np.ndarray
    ensemble_prediction: np.ndarray
    member_metrics: Mapping[int, Mapping[str, Any]]
    mask_seed: int
    expected_mask_checksum: str
    regenerated_mask_checksum: str
    ordered_gene_sha256: str
    ensemble_rule: str
    ensemble_rule_verified: bool
    prediction_scale: str
    target_preprocessing_uses_full_cell_library: bool
    oracle_true_library_size_used: bool
    graph_null_verified: bool = False
    degree_sequence_preserved: bool = False
    topology_preserved: bool = False
    permutation_seed: int | None = None


class ComparisonProvider(Protocol):
    """Model-specific checkpoint discovery and inference contract."""

    def audit(self) -> ComparisonAudit:
        """Return verified production and failure evidence."""

    def iter_evaluation_batches(self) -> Iterable[EvaluationBatch]:
        """Yield the complete frozen batch coverage exactly once."""

    def provenance(self) -> Mapping[str, Any]:
        """Return aggregate-only adapter and input provenance."""


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise GeneMAEComparisonError(f"{label} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeneMAEComparisonError(f"{label} must be finite numeric") from exc
    if not math.isfinite(result):
        raise GeneMAEComparisonError(f"{label} must be finite numeric")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise GeneMAEComparisonError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeneMAEComparisonError(f"{label} must be an integer") from exc
    if isinstance(value, float) and value != result:
        raise GeneMAEComparisonError(f"{label} must be an integer")
    return result


def _sha256_text(value: Any, *, label: str) -> str:
    result = str(value or "").lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise GeneMAEComparisonError(f"{label} must be a lowercase SHA-256")
    return result


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file without loading it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(name: str, value: np.ndarray) -> str:
    """Bind an array's semantic name, dtype, shape, and bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _assert_alias_safe_payload(value: Any, *, label: str) -> None:
    """Reject protected or row-level identifier fields before publication."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = (
                str(raw_key)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            if key in FORBIDDEN_REPORT_KEYS:
                raise GeneMAEComparisonError(
                    f"{label} contains prohibited identifier field {raw_key!r}"
                )
            _assert_alias_safe_payload(item, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_alias_safe_payload(item, label=label)


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise GeneMAEComparisonError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GeneMAEComparisonError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeneMAEComparisonError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise GeneMAEComparisonError(f"{label} must contain an object")
    return value


def _yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(json.dumps(__import__("yaml").safe_load(path.read_text())))
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise GeneMAEComparisonError(f"{label} is not valid YAML") from exc
    if not isinstance(value, dict):
        raise GeneMAEComparisonError(f"{label} must contain a mapping")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeneMAEComparisonError(f"{label} must be a mapping")
    return value


def _optional_json(
    root: Path, candidates: Sequence[str], *, label: str
) -> dict[str, Any]:
    present = [root / candidate for candidate in candidates if (root / candidate).is_file()]
    if len(present) != 1:
        raise GeneMAEComparisonError(
            f"{label} requires exactly one of {list(candidates)!r}"
        )
    return _strict_json(present[0], label=label)


def _state_dict_digest(state: Mapping[str, Any]) -> str:
    """Delegate to the runner's canonical replay/checkpoint digest."""

    from scripts.train.run_myjju_genemae_pooled import state_dict_sha256

    try:
        return state_dict_sha256(state)
    except (AttributeError, TypeError, ValueError) as exc:
        raise GeneMAEComparisonError(
            "checkpoint state mapping cannot be hashed canonically"
        ) from exc


def _registry_campaign_runs(
    registry: Registry, campaign_id: str
) -> list[dict[str, Any]]:
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT run_id, campaign_id, status, seed, attempt, retry_of,
                   artifact_path, parameter_count, duration_seconds,
                   peak_vram_gb, failure_category, config_json, created_at
            FROM runs
            WHERE campaign_id = ?
            ORDER BY created_at, run_id
            """,
            (campaign_id,),
        ).fetchall()
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            config = json.loads(str(item.pop("config_json")))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GeneMAEComparisonError(
                f"{campaign_id} registry contains invalid config JSON"
            ) from exc
        item["config"] = dict(_mapping(config, label="registered run config"))
        output.append(item)
    return output


def _execution_role(config: Mapping[str, Any]) -> str:
    metadata = config.get("metadata")
    if isinstance(metadata, Mapping):
        return str(metadata.get("execution_role", ""))
    return ""


def _configured_model_seed(config: Mapping[str, Any], fallback: Any) -> int:
    for container_name in ("experiment", "model", "metadata"):
        container = config.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for field in ("seed", "model_seed"):
            if field in container:
                return _integer(container[field], label=f"{container_name}.{field}")
    return _integer(fallback, label="registered seed")


def _configured_model_name(config: Mapping[str, Any]) -> str:
    model = config.get("model")
    if isinstance(model, Mapping):
        for field in ("name", "model_name", "family"):
            if model.get(field):
                return str(model[field])
    experiment = config.get("experiment")
    if isinstance(experiment, Mapping) and experiment.get("model_name"):
        return str(experiment["model_name"])
    return ""


def _resolve_artifact_root(
    artifact_path: Any, *, paths: ProjectPaths, run_id: str
) -> Path:
    if not isinstance(artifact_path, str) or not artifact_path:
        raise GeneMAEComparisonError(f"{run_id} has no registered artifact path")
    root = Path(artifact_path)
    if not root.is_absolute():
        root = paths.project_root / root
    root = root.resolve(strict=False)
    if not root.is_dir():
        raise GeneMAEComparisonError(f"{run_id} artifact root is absent")
    return root


def _load_checkpoint(
    path: Path, checkpoint_loader: Callable[..., Any]
) -> Mapping[str, Any]:
    try:
        payload = checkpoint_loader(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = checkpoint_loader(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise GeneMAEComparisonError("checkpoint payload must be a mapping")
    return payload


def discover_registered_genemae_production(
    *,
    registry: Registry,
    paths: ProjectPaths,
    bundle_verifier: Callable[..., Mapping[str, Any]] = verify_run_bundle,
    checkpoint_loader: Callable[..., Any] | None = None,
) -> tuple[
    tuple[RegisteredRunEvidence, ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    """Discover GeneMAE attempts and audit seven production bundles.

    Resource-pilot attempts remain visible but never enter production membership
    or checkpoint selection.  Production retries remain visible.  Only one
    terminal completed production attempt may occupy each seed slot, and every
    selected checkpoint must be the registered, checksum-verified
    ``checkpoints/last.ckpt`` artifact.
    """

    if checkpoint_loader is None:
        import torch

        checkpoint_loader = torch.load
    contract_path = (
        paths.project_root
        / "experiments"
        / "campaigns"
        / CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    if not contract_path.is_file() or sha256_file(contract_path) != FROZEN_CONTRACT_SHA256:
        raise GeneMAEComparisonError("frozen GeneMAE task contract checksum changed")
    source_audit_path = contract_path.with_name("external_source_audit.yaml")
    source_audit_sha = sha256_file(source_audit_path)

    rows = _registry_campaign_runs(registry, CAMPAIGN_ID)
    production: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pilot_inventory: list[dict[str, Any]] = []
    for row in rows:
        config = _mapping(row["config"], label="registered GeneMAE config")
        role = _execution_role(config)
        if role not in {"production", "resource_pilot"}:
            continue
        model_name = _configured_model_name(config)
        if model_name not in {GENEMAE, "myjju-genemae-source-fidelity"}:
            raise GeneMAEComparisonError(
                "GeneMAE production-like run has an unexpected model name"
            )
        seed = _configured_model_seed(config, row.get("seed"))
        attempt = _integer(row.get("attempt"), label="registered attempt")
        record = {
            "stage": "production" if role == "production" else "pilot",
            "model_key": GENEMAE,
            "seed": seed,
            "run_id": str(row["run_id"]),
            "attempt": attempt,
            "status": str(row["status"]),
            "retry_of": row.get("retry_of"),
            "failure_category": row.get("failure_category"),
            "selected_completed_attempt": False,
        }
        inventory.append(record)
        if role == "resource_pilot":
            pilot_inventory.append(record)
            if row.get("status") in {"failed", "pruned", "cancelled"}:
                failures.append(record)
            continue
        production.append({**row, "_seed": seed, "_attempt": attempt, "_record": record})

    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in production:
        seed = int(row["_seed"])
        if seed not in SEEDS:
            raise GeneMAEComparisonError(
                f"unexpected GeneMAE production seed {seed}"
            )
        by_seed.setdefault(seed, []).append(row)
    if set(by_seed) != set(SEEDS):
        raise GeneMAEComparisonError(
            "registry does not contain exact GeneMAE production seeds 0 through 6"
        )

    selected: list[dict[str, Any]] = []
    for seed in SEEDS:
        lineage = sorted(by_seed[seed], key=lambda row: int(row["_attempt"]))
        attempts = [int(row["_attempt"]) for row in lineage]
        if attempts != list(range(1, len(lineage) + 1)) or len(lineage) > 2:
            raise GeneMAEComparisonError(
                f"GeneMAE seed {seed} has an invalid retry lineage"
            )
        completed = [row for row in lineage if row.get("status") == "completed"]
        if len(completed) != 1 or completed[0] is not lineage[-1]:
            raise GeneMAEComparisonError(
                f"GeneMAE seed {seed} lacks one terminal completed attempt"
            )
        if any(
            row.get("status") not in {"failed", "pruned", "cancelled"}
            for row in lineage[:-1]
        ):
            raise GeneMAEComparisonError(
                f"GeneMAE seed {seed} has a nonterminal prior attempt"
            )
        previous_run_id: str | None = None
        for row in lineage:
            if row.get("retry_of") != previous_run_id:
                raise GeneMAEComparisonError(
                    f"GeneMAE seed {seed} retry linkage changed"
                )
            previous_run_id = str(row["run_id"])
        completed[0]["_record"]["selected_completed_attempt"] = True
        for row in lineage[:-1]:
            failures.append(row["_record"])
        selected.append(completed[0])

    evidence: list[RegisteredRunEvidence] = []
    for row in selected:
        seed = int(row["_seed"])
        run_id = str(row["run_id"])
        root = _resolve_artifact_root(
            row.get("artifact_path"), paths=paths, run_id=run_id
        )
        verification = bundle_verifier(root)
        if verification.get("valid") is not True or verification.get("status") != "success":
            raise GeneMAEComparisonError(f"{run_id} run bundle verification failed")
        artifact_issues = registry.verify_artifacts(run_id=run_id)
        if artifact_issues:
            raise GeneMAEComparisonError(
                f"{run_id} registry artifact verification failed"
            )
        config_path = root / "config.resolved.yaml"
        config = _yaml_mapping(config_path, label=f"{run_id} resolved config")
        if (
            _execution_role(config) != "production"
            or _configured_model_seed(config, seed) != seed
            or _configured_model_name(config)
            not in {GENEMAE, "myjju-genemae-source-fidelity"}
        ):
            raise GeneMAEComparisonError(
                f"{run_id} resolved production identity changed"
            )
        summary = _strict_json(root / "summary.json", label=f"{run_id} summary")
        if (
            summary.get("run_id") != run_id
            or summary.get("status") != "success"
            or summary.get("campaign_id") != CAMPAIGN_ID
            or summary.get("model_name") != GENEMAE
            or _integer(summary.get("model_seed"), label="summary model seed") != seed
            or _integer(summary.get("parameter_count"), label="summary parameter count")
            != EXPECTED_GENEMAE_PARAMETERS
            or _integer(
                summary.get("completed_global_epochs"),
                label="summary completed epochs",
            )
            != EXPECTED_GENEMAE_EPOCHS
            or _integer(summary.get("final_epoch"), label="summary final epoch")
            != EXPECTED_GENEMAE_FINAL_EPOCH
            or summary.get("checkpoint_role") != "last"
            or summary.get("primary_metric_name")
            != "fit/partial_gene/log1p_cp10k_masked_huber"
            or summary.get("generalization_estimate") is not False
        ):
            raise GeneMAEComparisonError(
                f"{run_id} summary violates the frozen production identity"
            )

        checkpoint_path = root / "checkpoints" / "last.ckpt"
        if not checkpoint_path.is_file():
            raise GeneMAEComparisonError(f"{run_id} lacks checkpoints/last.ckpt")
        checkpoint_sha = sha256_file(checkpoint_path)
        catalog = registry.list_checkpoint_catalog(
            run_id=run_id, role="last", limit=None
        )
        if len(catalog) != 1:
            raise GeneMAEComparisonError(
                f"{run_id} must have one cataloged last checkpoint"
            )
        catalog_row = catalog[0]
        catalog_path = Path(str(catalog_row.get("path")))
        if not catalog_path.is_absolute():
            catalog_path = paths.project_root / catalog_path
        if (
            catalog_path.resolve(strict=False) != checkpoint_path.resolve(strict=True)
            or catalog_row.get("sha256") != checkpoint_sha
            or catalog_row.get("verification_status") != "verified"
            or catalog_row.get("artifact_status") != "present"
            or catalog_row.get("role") != "last"
            or _integer(catalog_row.get("best_epoch"), label="catalog final epoch")
            != EXPECTED_GENEMAE_FINAL_EPOCH
        ):
            raise GeneMAEComparisonError(
                f"{run_id} checkpoint catalog metadata is invalid"
            )

        payload = _load_checkpoint(checkpoint_path, checkpoint_loader)
        state = payload.get("model_state_dict", payload.get("state_dict"))
        if not isinstance(state, Mapping):
            raise GeneMAEComparisonError(
                f"{run_id} checkpoint has no model state mapping"
            )
        state_sha = _state_dict_digest(state)
        declared_state_sha = _sha256_text(
            payload.get("state_dict_sha256"), label="checkpoint state checksum"
        )
        from .myjju_genemae import SOURCE_COMMIT, SOURCE_FILE_SHA256

        if (
            payload.get("run_id") != run_id
            or payload.get("model_name") != GENEMAE
            or _integer(payload.get("seed"), label="checkpoint seed") != seed
            or _integer(payload.get("final_epoch"), label="checkpoint final epoch")
            != EXPECTED_GENEMAE_FINAL_EPOCH
            or _integer(
                payload.get("parameter_count"), label="checkpoint parameter count"
            )
            != EXPECTED_GENEMAE_PARAMETERS
            or state_sha != declared_state_sha
            or payload.get("external_source_commit") != SOURCE_COMMIT
            or payload.get("external_source_file_sha256")
            != dict(SOURCE_FILE_SHA256)
            or payload.get("source_audit_sha256") != source_audit_sha
            or payload.get("source_fidelity_repair")
            != "assign_self_hidden_constructor_attribute"
            or payload.get("historical_weights_used") is not False
            or payload.get("target_scale") != TARGET_SCALE
            or _finite(
                payload.get("training_mask_rate"),
                label="checkpoint training mask rate",
            )
            != NATIVE_MASK_RATE
            or payload.get("runtime_config_sha256")
            != canonical_sha256(config)
        ):
            raise GeneMAEComparisonError(
                f"{run_id} checkpoint payload identity is invalid"
            )
        if (
            summary.get("checkpoint_path") != "checkpoints/last.ckpt"
            or summary.get("checkpoint_sha256") != checkpoint_sha
            or summary.get("state_dict_sha256") != state_sha
        ):
            raise GeneMAEComparisonError(
                f"{run_id} summary and checkpoint differ"
            )
        source_provenance = _strict_json(
            root / "provenance/external_source_audit.json",
            label=f"{run_id} source provenance",
        )
        if (
            source_provenance.get("contract_sha256")
            != FROZEN_CONTRACT_SHA256
            or source_provenance.get("source_audit_sha256")
            != source_audit_sha
            or source_provenance.get("external_source_commit")
            != SOURCE_COMMIT
            or source_provenance.get("external_source_file_sha256")
            != dict(SOURCE_FILE_SHA256)
        ):
            raise GeneMAEComparisonError(
                f"{run_id} source provenance changed"
            )
        graph_provenance = _strict_json(
            root / "provenance/tiled_graphs.json",
            label=f"{run_id} graph provenance",
        )
        mask_provenance = _strict_json(
            root / "provenance/fixed_evaluation_masks.json",
            label=f"{run_id} mask provenance",
        )
        if (
            payload.get("graph_bundle_sha256")
            != graph_provenance.get("graph_bundle_sha256")
            or payload.get("evaluation_mask_identities")
            != mask_provenance.get("masks")
            or mask_provenance.get("common_mask_rate")
            != COMMON_MASK_RATE
            or mask_provenance.get("native_mask_rate")
            != NATIVE_MASK_RATE
            or mask_provenance.get("replicates_per_rate") != len(REPLICATES)
        ):
            raise GeneMAEComparisonError(
                f"{run_id} graph or mask provenance differs from checkpoint"
            )

        convergence = _optional_json(
            root,
            ("diagnostics/convergence.json",),
            label=f"{run_id} convergence",
        )
        if (
            convergence.get("completed_epochs") != EXPECTED_GENEMAE_EPOCHS
            or convergence.get("expected_epochs") != EXPECTED_GENEMAE_EPOCHS
            or convergence.get("every_tile_once_each_epoch") is not True
            or convergence.get("all_losses_finite") is not True
            or convergence.get("all_gradients_finite") is not True
            or convergence.get("all_parameters_finite") is not True
            or convergence.get("all_global_epochs_completed") is not True
            or convergence.get("all_losses_and_gradients_finite") is not True
            or convergence.get("checkpoint_is_final_epoch_only") is not True
        ):
            raise GeneMAEComparisonError(
                f"{run_id} convergence audit is incomplete"
            )
        resources = _optional_json(
            root,
            ("diagnostics/resource.json",),
            label=f"{run_id} resources",
        )
        peak_vram = _finite(
            resources.get(
                "peak_allocated_vram_gib",
                resources.get("peak_vram_gib", row.get("peak_vram_gb")),
            ),
            label="peak VRAM",
        )
        peak_host = _finite(
            resources.get(
                "peak_host_memory_gib",
                resources.get("peak_rss_gib"),
            ),
            label="peak host memory",
        )
        duration = _finite(
            summary.get("duration_seconds", row.get("duration_seconds")),
            label="duration",
        )
        evidence.append(
            RegisteredRunEvidence(
                model_key=GENEMAE,
                seed=seed,
                run_id=run_id,
                attempt=int(row["_attempt"]),
                artifact_root=root,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                state_dict_sha256=state_sha,
                config_sha256=sha256_file(config_path),
                bundle_verified=True,
                registry_artifacts_verified=True,
                checkpoint_catalog_verified=True,
                parameter_count=EXPECTED_GENEMAE_PARAMETERS,
                completed_epochs=EXPECTED_GENEMAE_EPOCHS,
                final_epoch=EXPECTED_GENEMAE_FINAL_EPOCH,
                duration_seconds=duration,
                peak_vram_gib=peak_vram,
                peak_host_memory_gib=peak_host,
                convergence=convergence,
                resources=resources,
            )
        )
    return (
        tuple(sorted(evidence, key=lambda item: item.seed)),
        tuple(inventory),
        tuple(failures),
        tuple(pilot_inventory),
    )


def validate_comparison_audit(audit: ComparisonAudit) -> None:
    """Require exact seven-member coverage and verified immutable evidence."""

    if set(audit.members) != set(MODEL_KEYS):
        raise GeneMAEComparisonError(
            "comparison audit must contain GeneMAE, BAGM GAT, and BAGM self"
        )
    all_run_ids: set[str] = set()
    for model_key in MODEL_KEYS:
        members = audit.members[model_key]
        if len(members) != len(SEEDS):
            raise GeneMAEComparisonError(
                f"{model_key} does not contain exactly seven production members"
            )
        by_seed = {member.seed: member for member in members}
        if set(by_seed) != set(SEEDS) or len(by_seed) != len(members):
            raise GeneMAEComparisonError(
                f"{model_key} does not contain exact unique seeds 0 through 6"
            )
        for member in members:
            if member.model_key != model_key:
                raise GeneMAEComparisonError("member model key changed")
            if member.run_id in all_run_ids:
                raise GeneMAEComparisonError("production run ID was reused")
            all_run_ids.add(member.run_id)
            if not (
                member.bundle_verified
                and member.registry_artifacts_verified
                and member.checkpoint_catalog_verified
            ):
                raise GeneMAEComparisonError(
                    f"{member.run_id} lacks complete verification"
                )
            if not member.checkpoint_path.is_file():
                raise GeneMAEComparisonError(
                    f"{member.run_id} checkpoint is absent"
                )
            if sha256_file(member.checkpoint_path) != _sha256_text(
                member.checkpoint_sha256, label="member checkpoint checksum"
            ):
                raise GeneMAEComparisonError(
                    f"{member.run_id} checkpoint changed after audit"
                )
            if model_key == GENEMAE and (
                member.parameter_count != EXPECTED_GENEMAE_PARAMETERS
                or member.completed_epochs != EXPECTED_GENEMAE_EPOCHS
                or member.final_epoch != EXPECTED_GENEMAE_FINAL_EPOCH
            ):
                raise GeneMAEComparisonError(
                    f"{member.run_id} violates GeneMAE production dimensions"
                )
            for label, value in (
                ("duration", member.duration_seconds),
                ("peak VRAM", member.peak_vram_gib),
                ("peak host memory", member.peak_host_memory_gib),
            ):
                if _finite(value, label=label) < 0:
                    raise GeneMAEComparisonError(f"{label} cannot be negative")
    _assert_alias_safe_payload(
        audit.attempt_inventory, label="attempt inventory"
    )
    _assert_alias_safe_payload(
        audit.failure_inventory, label="failure inventory"
    )
    _assert_alias_safe_payload(audit.pilot_inventory, label="pilot inventory")
    _assert_alias_safe_payload(audit.provenance or {}, label="audit provenance")
    provenance = _mapping(
        audit.provenance, label="comparison audit provenance"
    )
    bagm_context = _mapping(
        provenance.get("current_bagm_whole_node_context"),
        label="current BAGM whole-node context",
    )
    for field, expected in EXPECTED_CURRENT_BAGM_CONTEXT.items():
        observed = bagm_context.get(field)
        if isinstance(expected, float):
            if not math.isclose(
                _finite(observed, label=f"current BAGM {field}"),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise GeneMAEComparisonError(
                    f"current BAGM contextual {field} changed"
                )
        elif observed != expected:
            raise GeneMAEComparisonError(
                f"current BAGM contextual {field} changed"
            )
    if (
        bagm_context.get("context_only_not_ranked_against_genemae") is not True
        or bagm_context.get("graph_gate_passed") is not True
        or bagm_context.get("overall_campaign_outcome") != "negative"
        or set(bagm_context.get("failed_gates", ()))
        != {"pooled_data_gate", "representation_gate"}
    ):
        raise GeneMAEComparisonError(
            "current BAGM whole-node context lost its estimand or negative outcome"
        )
    source_context = _mapping(
        provenance.get("source_report_historical_context"),
        label="source-report historical context",
    )
    source_metrics = _mapping(
        source_context.get("metrics"), label="source historical metrics"
    )
    for cohort_key, expected_metrics in EXPECTED_SOURCE_HISTORICAL_METRICS.items():
        observed_metrics = _mapping(
            source_metrics.get(cohort_key),
            label=f"source historical {cohort_key}",
        )
        for metric, expected in expected_metrics.items():
            if not math.isclose(
                _finite(
                    observed_metrics.get(metric),
                    label=f"source historical {cohort_key} {metric}",
                ),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise GeneMAEComparisonError(
                    f"source historical {cohort_key} {metric} changed"
                )
    if (
        source_context.get("context_only_not_comparable") is not True
        or source_context.get("historical_weights_available") is not False
        or source_context.get("checkpoint_selection_leakage") is not True
        or len(
            _sha256_text(
                source_context.get("source_audit_sha256"),
                label="source historical audit checksum",
            )
        )
        != 64
    ):
        raise GeneMAEComparisonError(
            "source historical context lost its non-comparability caveat"
        )


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if x64.size < 2 or y64.size != x64.size:
        return None
    x_centered = x64 - float(np.mean(x64, dtype=np.float64))
    y_centered = y64 - float(np.mean(y64, dtype=np.float64))
    denominator = math.sqrt(
        float(np.dot(x_centered, x_centered))
        * float(np.dot(y_centered, y_centered))
    )
    if denominator <= 0 or not math.isfinite(denominator):
        return None
    value = float(np.dot(x_centered, y_centered) / denominator)
    return max(-1.0, min(1.0, value))


def _axis_correlations(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, *, axis: int
) -> np.ndarray:
    width = target.shape[1 - axis]
    values = np.full(width, np.nan, dtype=np.float64)
    for index in range(width):
        selected = mask[:, index] if axis == 0 else mask[index, :]
        x = target[:, index][selected] if axis == 0 else target[index, :][selected]
        y = (
            prediction[:, index][selected]
            if axis == 0
            else prediction[index, :][selected]
        )
        correlation = _correlation(x, y)
        if correlation is not None:
            values[index] = correlation
    return values


def masked_regression_metrics(
    target: Any, prediction: Any, mask: Any
) -> dict[str, Any]:
    """Compute all frozen metrics on masked full-cell log1p(CP10k) entries."""

    from .myjju_genemae import MaskedRegressionAccumulator

    truth = np.asarray(target)
    estimate = np.asarray(prediction)
    selected = np.asarray(mask)
    if (
        truth.ndim != 2
        or estimate.shape != truth.shape
        or selected.shape != truth.shape
        or selected.dtype != np.bool_
        or truth.shape[1] != EXPECTED_GENE_COUNT
    ):
        raise GeneMAEComparisonError(
            "metrics require aligned [cells,1000] target/prediction/bool mask"
        )
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise GeneMAEComparisonError("metric inputs must be finite")
    n_masked = int(np.count_nonzero(selected))
    if n_masked == 0:
        raise GeneMAEComparisonError("metric mask is empty")
    accumulator = MaskedRegressionAccumulator(EXPECTED_GENE_COUNT)
    accumulator.update(truth, estimate, selected)
    raw = accumulator.finalize()
    result: dict[str, Any] = {
        metric: (
            None
            if isinstance(raw[metric], (float, np.floating))
            and not math.isfinite(float(raw[metric]))
            else float(raw[metric])
        )
        for metric in METRIC_NAMES
    }
    result.update(
        {
            "n_masked": int(raw["n_masked"]),
            "defined_gene_correlations": int(raw["n_valid_genes"]),
            "defined_cell_correlations": int(raw["n_valid_cells"]),
        }
    )
    for name in ("masked_huber", "masked_mse", "masked_mae"):
        _finite(result[name], label=name)
    return result


def _validated_metrics(
    value: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        raw = value.get(metric)
        if raw is None:
            if metric not in CORRELATION_METRICS and metric != "masked_r2":
                raise GeneMAEComparisonError(f"{label} lacks {metric}")
            result[metric] = None
        else:
            result[metric] = _finite(raw, label=f"{label} {metric}")
    n_masked = _integer(value.get("n_masked"), label=f"{label} n_masked")
    if n_masked < 1:
        raise GeneMAEComparisonError(f"{label} n_masked must be positive")
    result["n_masked"] = n_masked
    for count_name in ("defined_gene_correlations", "defined_cell_correlations"):
        if count_name in value:
            count = _integer(value[count_name], label=f"{label} {count_name}")
            if count < 0:
                raise GeneMAEComparisonError(
                    f"{label} {count_name} cannot be negative"
                )
            result[count_name] = count
    return result


def _expected_batch_keys() -> set[tuple[str, str, float, int, str]]:
    result: set[tuple[str, str, float, int, str]] = set()
    for alias in ALIASES:
        for replicate in REPLICATES:
            for rate in (COMMON_MASK_RATE, NATIVE_MASK_RATE):
                for condition in ("observed", "node_label_permuted"):
                    result.add((GENEMAE, alias, rate, replicate, condition))
            for model_key in (BAGM_GAT, BAGM_SELF):
                result.add(
                    (
                        model_key,
                        alias,
                        COMMON_MASK_RATE,
                        replicate,
                        "observed",
                    )
                )
    return result


def _batch_key(batch: EvaluationBatch) -> tuple[str, str, float, int, str]:
    return (
        batch.model_key,
        batch.core_alias,
        float(batch.mask_rate),
        int(batch.replicate),
        batch.graph_condition,
    )


def _validate_batch(batch: EvaluationBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = _batch_key(batch)
    if key not in _expected_batch_keys():
        raise GeneMAEComparisonError(f"unexpected evaluation batch {key!r}")
    target = np.asarray(batch.target)
    mask = np.asarray(batch.mask)
    prediction = np.asarray(batch.ensemble_prediction)
    if (
        target.ndim != 2
        or target.shape[1] != EXPECTED_GENE_COUNT
        or mask.shape != target.shape
        or prediction.shape != target.shape
        or mask.dtype != np.bool_
    ):
        raise GeneMAEComparisonError(f"{key!r} has invalid array shapes or mask dtype")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise GeneMAEComparisonError(f"{key!r} contains non-finite arrays")
    if np.any(target < 0):
        raise GeneMAEComparisonError(f"{key!r} target cannot be negative")
    density = float(np.mean(mask))
    tolerance = max(1.0 / mask.size, 0.002)
    if abs(density - batch.mask_rate) > tolerance:
        raise GeneMAEComparisonError(
            f"{key!r} mask density differs from its declared rate"
        )
    expected_mask = _sha256_text(
        batch.expected_mask_checksum, label="expected mask checksum"
    )
    regenerated_mask = _sha256_text(
        batch.regenerated_mask_checksum, label="regenerated mask checksum"
    )
    if expected_mask != regenerated_mask:
        raise GeneMAEComparisonError(
            f"{key!r} regenerated mask checksum does not match production"
        )
    _sha256_text(batch.ordered_gene_sha256, label="ordered gene checksum")
    if batch.prediction_scale != TARGET_SCALE:
        raise GeneMAEComparisonError(f"{key!r} prediction scale changed")
    if batch.target_preprocessing_uses_full_cell_library is not True:
        raise GeneMAEComparisonError(
            f"{key!r} did not attest full-cell pre-mask normalization"
        )
    expected_rule = (
        GENEMAE_ENSEMBLE_RULE
        if batch.model_key == GENEMAE
        else BAGM_ENSEMBLE_RULE
    )
    if (
        batch.ensemble_rule != expected_rule
        or batch.ensemble_rule_verified is not True
    ):
        raise GeneMAEComparisonError(
            f"{key!r} did not verify the canonical prediction ensemble"
        )
    if batch.model_key in {BAGM_GAT, BAGM_SELF}:
        if batch.oracle_true_library_size_used is not True:
            raise GeneMAEComparisonError(
                f"{key!r} BAGM prediction lacks oracle-scale attestation"
            )
    elif batch.oracle_true_library_size_used is not False:
        raise GeneMAEComparisonError(
            f"{key!r} GeneMAE cannot claim BAGM oracle count rescaling"
        )
    if batch.graph_condition == "node_label_permuted":
        if (
            batch.model_key != GENEMAE
            or batch.graph_null_verified is not True
            or batch.degree_sequence_preserved is not True
            or batch.topology_preserved is not True
            or batch.permutation_seed is None
        ):
            raise GeneMAEComparisonError(
                f"{key!r} graph null is not fully verified"
            )
    elif any(
        (
            batch.graph_null_verified,
            batch.degree_sequence_preserved,
            batch.topology_preserved,
        )
    ):
        raise GeneMAEComparisonError(
            f"{key!r} observed graph is mislabeled as a null"
        )
    if set(batch.member_metrics) != set(SEEDS):
        raise GeneMAEComparisonError(
            f"{key!r} lacks exact seven-member metric coverage"
        )
    n_masked = int(np.count_nonzero(mask))
    for seed in SEEDS:
        metrics = _validated_metrics(
            batch.member_metrics[seed], label=f"{key!r} seed {seed}"
        )
        if metrics["n_masked"] != n_masked:
            raise GeneMAEComparisonError(
                f"{key!r} seed {seed} was not scored on the same mask"
            )
    return target, mask, prediction


def _row_metrics(
    *,
    identity: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {**identity, **_validated_metrics(metrics, label=str(identity))}


def _mean_defined(values: Sequence[Any], *, label: str) -> float | None:
    numeric = [
        _finite(value, label=label)
        for value in values
        if value is not None
    ]
    return (
        None
        if not numeric
        else float(np.mean(numeric, dtype=np.float64))
    )


def collapse_mask_replicates(
    rows: Sequence[Mapping[str, Any]], *, identity_fields: Sequence[str]
) -> list[dict[str, Any]]:
    """Average the three technical masks before any core or seed aggregate."""

    grouped: dict[tuple[Any, ...], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in identity_fields)
        replicate = _integer(row.get("replicate"), label="mask replicate")
        if replicate not in REPLICATES or replicate in grouped.setdefault(key, {}):
            raise GeneMAEComparisonError(
                "metric rows contain an invalid or duplicate mask replicate"
            )
        grouped[key][replicate] = row
    output: list[dict[str, Any]] = []
    for key, by_replicate in sorted(grouped.items(), key=lambda item: str(item[0])):
        if set(by_replicate) != set(REPLICATES):
            raise GeneMAEComparisonError(
                f"metric group {key!r} lacks all three mask replicates"
            )
        item = {
            field: value for field, value in zip(identity_fields, key, strict=True)
        }
        item["mask_replicates"] = len(REPLICATES)
        for metric in METRIC_NAMES:
            item[metric] = _mean_defined(
                [by_replicate[index].get(metric) for index in REPLICATES],
                label=f"{key!r} {metric}",
            )
        item["n_masked_mean"] = float(
            np.mean(
                [
                    _integer(
                        by_replicate[index].get("n_masked"),
                        label=f"{key!r} n_masked",
                    )
                    for index in REPLICATES
                ],
                dtype=np.float64,
            )
        )
        output.append(item)
    return output


def equal_core_aggregates(
    rows: Sequence[Mapping[str, Any]], *, identity_fields: Sequence[str]
) -> list[dict[str, Any]]:
    """Average core rows with exactly equal weight."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in identity_fields)
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items(), key=lambda item: str(item[0])):
        aliases = [str(row.get("core_alias")) for row in selected]
        if len(aliases) != len(ALIASES) or set(aliases) != set(ALIASES):
            raise GeneMAEComparisonError(
                f"equal-core group {key!r} lacks exact ten-core coverage"
            )
        item = {
            field: value for field, value in zip(identity_fields, key, strict=True)
        }
        item.update({"core_count": len(ALIASES), "core_weighting": "equal"})
        for metric in METRIC_NAMES:
            item[metric] = _mean_defined(
                [row.get(metric) for row in selected],
                label=f"{key!r} {metric}",
            )
        output.append(item)
    return output


def _indexed(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in result:
            raise GeneMAEComparisonError(f"duplicate row identity {key!r}")
        result[key] = row
    return result


def relative_improvement(
    baseline: Any, candidate: Any, *, label: str
) -> float:
    reference = _finite(baseline, label=f"{label} baseline")
    value = _finite(candidate, label=f"{label} candidate")
    if reference <= 0:
        raise GeneMAEComparisonError(
            f"{label} relative-improvement baseline must be positive"
        )
    return (reference - value) / reference


def _control_contrast_rows(
    equal_core_rows: Sequence[Mapping[str, Any]],
    *,
    current_bagm_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose metric- and estimand-dependent control results deterministically."""

    indexed = _indexed(
        equal_core_rows, ("model_key", "mask_rate", "graph_condition")
    )
    gene = indexed[(GENEMAE, COMMON_MASK_RATE, "observed")]
    zero = indexed[("all_zero", COMMON_MASK_RATE, "observed")]
    gat = indexed[(BAGM_GAT, COMMON_MASK_RATE, "observed")]
    matched_self = indexed[(BAGM_SELF, COMMON_MASK_RATE, "observed")]

    gene_huber_gain = relative_improvement(
        zero["masked_huber"],
        gene["masked_huber"],
        label="GeneMAE versus all-zero Huber",
    )
    gene_mae_gain = relative_improvement(
        zero["masked_mae"],
        gene["masked_mae"],
        label="GeneMAE versus all-zero MAE",
    )
    gene_huber_favored = gene_huber_gain > 0.0
    gene_mae_favored = gene_mae_gain > 0.0

    gat_huber_gain = relative_improvement(
        matched_self["masked_huber"],
        gat["masked_huber"],
        label="BAGM GAT versus matched-self partial-gene Huber",
    )
    return [
        {
            "contrast": "genemae_vs_all_zero_common_partial_gene",
            "estimand": "held_in_partial_gene_reconstruction",
            "mask_rate": COMMON_MASK_RATE,
            "candidate_model": GENEMAE,
            "reference_model": "all_zero",
            "candidate_huber": gene["masked_huber"],
            "reference_huber": zero["masked_huber"],
            "candidate_huber_relative_improvement": gene_huber_gain,
            "huber_favors_candidate": gene_huber_favored,
            "candidate_mae": gene["masked_mae"],
            "reference_mae": zero["masked_mae"],
            "candidate_mae_relative_improvement": gene_mae_gain,
            "mae_favors_candidate": gene_mae_favored,
            "metric_rank_agreement": (
                gene_huber_favored == gene_mae_favored
            ),
            "metric_dependent_conclusion": (
                gene_huber_favored != gene_mae_favored
            ),
        },
        {
            "contrast": "bagm_gat_vs_matched_self_common_partial_gene",
            "estimand": "held_in_partial_gene_reconstruction",
            "mask_rate": COMMON_MASK_RATE,
            "candidate_model": BAGM_GAT,
            "reference_model": BAGM_SELF,
            "candidate_huber": gat["masked_huber"],
            "reference_huber": matched_self["masked_huber"],
            "candidate_huber_relative_improvement": gat_huber_gain,
            "candidate_huber_relative_loss_increase": -gat_huber_gain,
            "huber_favors_candidate": gat_huber_gain > 0.0,
            "separate_whole_node_graph_gate_passed": (
                current_bagm_context.get("graph_gate_passed") is True
            ),
            "separate_whole_node_estimand": current_bagm_context.get(
                "estimand"
            ),
            "estimands_are_distinct": True,
            "architecture_level_conclusion_permitted": False,
        },
    ]


def _control_negative_results(
    control_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    by_contrast = {
        str(row.get("contrast")): row for row in control_rows
    }
    if len(by_contrast) != len(control_rows):
        raise GeneMAEComparisonError("control contrasts contain duplicates")
    results: list[str] = []
    zero = by_contrast["genemae_vs_all_zero_common_partial_gene"]
    if zero.get("metric_dependent_conclusion") is True:
        huber_gain = _finite(
            zero.get("candidate_huber_relative_improvement"),
            label="GeneMAE all-zero Huber contrast",
        )
        mae_gain = _finite(
            zero.get("candidate_mae_relative_improvement"),
            label="GeneMAE all-zero MAE contrast",
        )
        if huber_gain > 0.0 and mae_gain < 0.0:
            results.append(
                "Metric dependence on the common 20% task: GeneMAE was "
                f"{100.0 * huber_gain:.2f}% better than all-zero by Huber but "
                f"{100.0 * -mae_gain:.2f}% worse by MAE; the control "
                "conclusion is not metric-robust."
            )
        else:
            results.append(
                "Metric dependence on the common 20% task: GeneMAE and "
                "all-zero reverse rank between Huber and MAE; the control "
                "conclusion is not metric-robust."
            )
    bagm = by_contrast[
        "bagm_gat_vs_matched_self_common_partial_gene"
    ]
    loss_increase = _finite(
        bagm.get("candidate_huber_relative_loss_increase"),
        label="BAGM partial-gene GAT contrast",
    )
    if loss_increase > 0.0:
        results.append(
            "On the common partial-gene task, BAGM GAT Huber was "
            f"{100.0 * loss_increase:.2f}% higher than matched self. Its "
            "separate whole-node graph gate passed; this is task- and "
            "estimand-specific evidence, not an architecture-level result."
        )
    return results


def _gate_record(
    name: str, *, passed: bool, observed: Mapping[str, Any], threshold: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "gate": name,
        "passed": bool(passed),
        "observed": dict(observed),
        "threshold": dict(threshold),
    }


def evaluate_frozen_gates(
    *,
    core_rows: Sequence[Mapping[str, Any]],
    equal_core_rows: Sequence[Mapping[str, Any]],
    member_equal_core_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate the frozen primary, baseline, and graph-use gates."""

    core = _indexed(
        core_rows,
        ("model_key", "core_alias", "mask_rate", "graph_condition"),
    )
    equal = _indexed(
        equal_core_rows, ("model_key", "mask_rate", "graph_condition")
    )
    members = _indexed(
        member_equal_core_rows,
        ("model_key", "seed", "mask_rate", "graph_condition"),
    )
    gen_key = (GENEMAE, COMMON_MASK_RATE, "observed")
    gat_key = (BAGM_GAT, COMMON_MASK_RATE, "observed")
    null_key = (GENEMAE, COMMON_MASK_RATE, "node_label_permuted")
    gene = equal[gen_key]
    gat = equal[gat_key]
    null = equal[null_key]
    comparisons: list[dict[str, Any]] = []
    primary_core_favored = 0
    baseline_core_favored = 0
    graph_core_favored = 0
    for alias in ALIASES:
        gene_core = core[(GENEMAE, alias, COMMON_MASK_RATE, "observed")]
        gat_core = core[(BAGM_GAT, alias, COMMON_MASK_RATE, "observed")]
        null_core = core[
            (GENEMAE, alias, COMMON_MASK_RATE, "node_label_permuted")
        ]
        mean_core = core[
            ("per_core_gene_mean", alias, COMMON_MASK_RATE, "observed")
        ]
        primary_gain = relative_improvement(
            gat_core[PRIMARY_METRIC],
            gene_core[PRIMARY_METRIC],
            label=f"primary {alias}",
        )
        baseline_gain = relative_improvement(
            mean_core[PRIMARY_METRIC],
            gene_core[PRIMARY_METRIC],
            label=f"baseline {alias}",
        )
        graph_gain = relative_improvement(
            null_core[PRIMARY_METRIC],
            gene_core[PRIMARY_METRIC],
            label=f"graph use {alias}",
        )
        primary_favored = _finite(
            gene_core[PRIMARY_METRIC], label=f"primary {alias} GeneMAE Huber"
        ) < _finite(
            gat_core[PRIMARY_METRIC], label=f"primary {alias} BAGM GAT Huber"
        )
        baseline_favored = _finite(
            gene_core[PRIMARY_METRIC], label=f"baseline {alias} GeneMAE Huber"
        ) < _finite(
            mean_core[PRIMARY_METRIC], label=f"baseline {alias} gene-mean Huber"
        )
        graph_favored = _finite(
            gene_core[PRIMARY_METRIC], label=f"graph {alias} observed Huber"
        ) < _finite(
            null_core[PRIMARY_METRIC], label=f"graph {alias} permuted Huber"
        )
        primary_core_favored += int(primary_favored)
        baseline_core_favored += int(baseline_favored)
        graph_core_favored += int(graph_favored)
        comparisons.append(
            {
                "core_alias": alias,
                "genemae_huber": gene_core[PRIMARY_METRIC],
                "bagm_gat_huber": gat_core[PRIMARY_METRIC],
                "per_core_gene_mean_huber": mean_core[PRIMARY_METRIC],
                "genemae_permuted_huber": null_core[PRIMARY_METRIC],
                "genemae_vs_bagm_gat_relative_improvement": primary_gain,
                "genemae_vs_gene_mean_relative_improvement": baseline_gain,
                "genemae_observed_vs_permuted_relative_improvement": graph_gain,
                "genemae_favored_vs_bagm_gat": primary_favored,
                "genemae_favored_vs_gene_mean": baseline_favored,
                "observed_graph_favored_vs_permuted": graph_favored,
            }
        )

    favoring_seed_pairs = 0
    seed_pair_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        gene_seed = members[
            (GENEMAE, seed, COMMON_MASK_RATE, "observed")
        ]
        gat_seed = members[
            (BAGM_GAT, seed, COMMON_MASK_RATE, "observed")
        ]
        gain = relative_improvement(
            gat_seed[PRIMARY_METRIC],
            gene_seed[PRIMARY_METRIC],
            label=f"same-numbered seed {seed}",
        )
        favored = gene_seed[PRIMARY_METRIC] < gat_seed[PRIMARY_METRIC]
        favoring_seed_pairs += int(favored)
        seed_pair_rows.append(
            {
                "seed": seed,
                "genemae_equal_core_huber": gene_seed[PRIMARY_METRIC],
                "bagm_gat_equal_core_huber": gat_seed[PRIMARY_METRIC],
                "genemae_relative_improvement": gain,
                "genemae_favored": favored,
            }
        )

    primary_gain = relative_improvement(
        gat[PRIMARY_METRIC], gene[PRIMARY_METRIC], label="primary equal core"
    )
    baseline = equal[
        ("per_core_gene_mean", COMMON_MASK_RATE, "observed")
    ]
    baseline_gain = relative_improvement(
        baseline[PRIMARY_METRIC],
        gene[PRIMARY_METRIC],
        label="baseline equal core",
    )
    graph_gain = relative_improvement(
        null[PRIMARY_METRIC], gene[PRIMARY_METRIC], label="graph equal core"
    )
    mae_nonworse = _finite(
        gene["masked_mae"], label="GeneMAE equal-core MAE"
    ) <= _finite(gat["masked_mae"], label="BAGM GAT equal-core MAE")
    gene_corr = gene.get("gene_pearson_mean")
    gat_corr = gat.get("gene_pearson_mean")
    gene_corr_nonworse = (
        gene_corr is not None
        and gat_corr is not None
        and _finite(gene_corr, label="GeneMAE gene Pearson")
        >= _finite(gat_corr, label="BAGM GAT gene Pearson")
    )
    primary_passed = (
        primary_gain >= 0.02
        and primary_core_favored >= 8
        and favoring_seed_pairs >= 5
        and mae_nonworse
        and gene_corr_nonworse
    )
    baseline_passed = baseline_gain >= 0.02 and baseline_core_favored >= 8
    graph_passed = graph_gain >= 0.02 and graph_core_favored >= 8
    gates = [
        _gate_record(
            "primary_comparison",
            passed=primary_passed,
            observed={
                "mean_huber_relative_improvement": primary_gain,
                "genemae_favoring_cores": primary_core_favored,
                "genemae_favoring_seed_pairs": favoring_seed_pairs,
                "masked_mae_nonworse": mae_nonworse,
                "mean_gene_pearson_nonworse": gene_corr_nonworse,
            },
            threshold={
                "mean_huber_relative_improvement_minimum": 0.02,
                "minimum_genemae_favoring_cores": 8,
                "minimum_genemae_favoring_seed_pairs": 5,
                "masked_mae_nonworse": True,
                "mean_gene_pearson_nonworse": True,
            },
        ),
        _gate_record(
            "baseline",
            passed=baseline_passed,
            observed={
                "mean_huber_relative_improvement": baseline_gain,
                "genemae_favoring_cores": baseline_core_favored,
            },
            threshold={
                "mean_huber_relative_improvement_minimum": 0.02,
                "minimum_genemae_favoring_cores": 8,
            },
        ),
        _gate_record(
            "graph_use",
            passed=graph_passed,
            observed={
                "mean_huber_relative_improvement": graph_gain,
                "unpermuted_favoring_cores": graph_core_favored,
            },
            threshold={
                "mean_huber_relative_improvement_minimum": 0.02,
                "minimum_unpermuted_favoring_cores": 8,
            },
        ),
    ]
    return gates, comparisons + seed_pair_rows


def _core_effect_distribution_rows(
    comparison_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize coupled-core effects descriptively without inferential CIs."""

    core_rows = [
        row for row in comparison_rows if row.get("core_alias") is not None
    ]
    aliases = [str(row.get("core_alias")) for row in core_rows]
    if len(aliases) != len(ALIASES) or set(aliases) != set(ALIASES):
        raise GeneMAEComparisonError(
            "core-effect distribution lacks exact ten-core coverage"
        )
    definitions = (
        (
            "genemae_vs_bagm_gat",
            "genemae_vs_bagm_gat_relative_improvement",
            "genemae_favored_vs_bagm_gat",
        ),
        (
            "genemae_vs_target_derived_per_core_gene_mean",
            "genemae_vs_gene_mean_relative_improvement",
            "genemae_favored_vs_gene_mean",
        ),
        (
            "genemae_observed_vs_node_label_permuted",
            "genemae_observed_vs_permuted_relative_improvement",
            "observed_graph_favored_vs_permuted",
        ),
    )
    output: list[dict[str, Any]] = []
    for contrast, effect_field, favor_field in definitions:
        effects = np.asarray(
            [
                _finite(row.get(effect_field), label=f"{contrast} core effect")
                for row in core_rows
            ],
            dtype=np.float64,
        )
        favored = [bool(row.get(favor_field)) for row in core_rows]
        if favored != [bool(effect > 0.0) for effect in effects]:
            raise GeneMAEComparisonError(
                f"{contrast} favor indicators differ from strictly lower Huber"
            )
        output.append(
            {
                "contrast": contrast,
                "effect": "relative_masked_huber_improvement",
                "n_cores": len(ALIASES),
                "median": float(np.median(effects)),
                "minimum": float(np.min(effects)),
                "maximum": float(np.max(effects)),
                "favoring_core_count": int(sum(favored)),
                "favor_definition": "candidate_masked_huber_strictly_lower",
                "population_confidence_interval": None,
                "bootstrap_used": False,
                "inference": (
                    "descriptive_only_no_population_interval_because_all_cores_"
                    "are_held_in_and_coupled_by_shared_fitted_weights"
                ),
            }
        )
    return output


def _maximum_conclusion(
    *, primary_passed: bool, failed_gates: Sequence[str]
) -> str:
    if primary_passed:
        conclusion = (
            "The reproduced GeneMAE ensemble met the frozen descriptive "
            "common-task comparison gate against the current BAGM GAT ensemble "
            "on held-in 20% partial-gene reconstruction across ten "
            "adjacent-normal core aliases. This supports only a comparison of "
            "the two end-to-end trained systems, not an isolated architecture "
            "effect."
        )
    else:
        conclusion = (
            "The technically complete exploratory comparison did not meet the "
            "frozen primary GeneMAE-versus-current-BAGM gate; no common-task "
            "GeneMAE advantage is claimed."
        )
    non_primary_failures = [
        name for name in failed_gates if name in {"baseline", "graph_use"}
    ]
    if non_primary_failures:
        conclusion += (
            " Non-primary frozen gates failed: "
            + ", ".join(non_primary_failures)
            + "."
        )
        if "baseline" in non_primary_failures:
            conclusion += (
                " GeneMAE did not establish the prespecified advantage over "
                "the target-derived per-core gene-mean oracle."
            )
        if "graph_use" in non_primary_failures:
            conclusion += (
                " The graph-null result does not support graph-specific "
                "predictive gain."
            )
    return conclusion


def _resource_rows(audit: ComparisonAudit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        for member in sorted(audit.members[model_key], key=lambda item: item.seed):
            rows.append(
                {
                    "model_key": model_key,
                    "seed": member.seed,
                    "run_id": member.run_id,
                    "attempt": member.attempt,
                    "duration_seconds": member.duration_seconds,
                    "duration_hours": member.duration_seconds / 3600.0,
                    "peak_vram_gib": member.peak_vram_gib,
                    "peak_host_memory_gib": member.peak_host_memory_gib,
                    "parameter_count": member.parameter_count,
                    "completed_epochs": member.completed_epochs,
                    "final_epoch": member.final_epoch,
                    "failed_attempts_before_completion": sum(
                        1
                        for row in audit.failure_inventory
                        if row.get("model_key") == model_key
                        and row.get("seed") == member.seed
                        and row.get("stage") == "production"
                    ),
                }
            )
    return rows


def _run_audit_rows(audit: ComparisonAudit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        for member in sorted(audit.members[model_key], key=lambda item: item.seed):
            rows.append(
                {
                    "model_key": model_key,
                    "seed": member.seed,
                    "run_id": member.run_id,
                    "attempt": member.attempt,
                    "bundle_verified": member.bundle_verified,
                    "registry_artifacts_verified": member.registry_artifacts_verified,
                    "checkpoint_catalog_verified": member.checkpoint_catalog_verified,
                    "checkpoint_role": "last",
                    "checkpoint_file_sha256": member.checkpoint_sha256,
                    "state_dict_sha256": member.state_dict_sha256,
                    "config_sha256": member.config_sha256,
                    "parameter_count": member.parameter_count,
                    "completed_epochs": member.completed_epochs,
                    "final_epoch": member.final_epoch,
                }
            )
    return rows


def _seed_variability_rows(
    member_equal_core_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[Mapping[str, Any]]] = {}
    for row in member_equal_core_rows:
        grouped.setdefault(
            (
                str(row["model_key"]),
                float(row["mask_rate"]),
                str(row["graph_condition"]),
            ),
            [],
        ).append(row)
    output: list[dict[str, Any]] = []
    for (model_key, rate, condition), selected in sorted(grouped.items()):
        if {int(row["seed"]) for row in selected} != set(SEEDS):
            raise GeneMAEComparisonError(
                f"seed variability lacks seven seeds for {(model_key, rate, condition)}"
            )
        item: dict[str, Any] = {
            "model_key": model_key,
            "mask_rate": rate,
            "graph_condition": condition,
            "seed_count": len(SEEDS),
            "seeds_are_biological_replicates": False,
        }
        for metric in METRIC_NAMES:
            values = [
                _finite(row[metric], label=f"seed variability {metric}")
                for row in selected
                if row.get(metric) is not None
            ]
            item[f"{metric}_mean"] = (
                None if not values else float(np.mean(values, dtype=np.float64))
            )
            item[f"{metric}_sd"] = (
                None if not values else float(np.std(values, ddof=0))
            )
            item[f"{metric}_min"] = None if not values else float(min(values))
            item[f"{metric}_max"] = None if not values else float(max(values))
        output.append(item)
    return output


def analyze_provider(
    provider: ComparisonProvider,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Consume verified batches, compute tables, and evaluate all frozen gates."""

    audit = provider.audit()
    validate_comparison_audit(audit)
    seen: set[tuple[str, str, float, int, str]] = set()
    identities: dict[tuple[str, float, int], dict[str, str]] = {}
    ensemble_replicate_rows: list[dict[str, Any]] = []
    member_replicate_rows: list[dict[str, Any]] = []
    baseline_replicate_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []

    for batch in provider.iter_evaluation_batches():
        key = _batch_key(batch)
        if key in seen:
            raise GeneMAEComparisonError(f"duplicate evaluation batch {key!r}")
        seen.add(key)
        target, mask, prediction = _validate_batch(batch)
        target_sha = array_sha256("target_log1p_cp10k", target)
        mask_sha = array_sha256("entry_mask", mask)
        identity_key = (batch.core_alias, float(batch.mask_rate), batch.replicate)
        identity = identities.setdefault(
            identity_key,
            {
                "expected_mask_checksum": batch.expected_mask_checksum,
                "regenerated_mask_checksum": batch.regenerated_mask_checksum,
                "mask_array_sha256": mask_sha,
                "target_array_sha256": target_sha,
                "ordered_gene_sha256": batch.ordered_gene_sha256,
            },
        )
        if identity != {
            "expected_mask_checksum": batch.expected_mask_checksum,
            "regenerated_mask_checksum": batch.regenerated_mask_checksum,
            "mask_array_sha256": mask_sha,
            "target_array_sha256": target_sha,
            "ordered_gene_sha256": batch.ordered_gene_sha256,
        }:
            raise GeneMAEComparisonError(
                f"models do not share an identical target/mask for {identity_key!r}"
            )

        row_identity = {
            "model_key": batch.model_key,
            "core_alias": batch.core_alias,
            "mask_rate": float(batch.mask_rate),
            "replicate": batch.replicate,
            "graph_condition": batch.graph_condition,
        }
        ensemble_metrics = masked_regression_metrics(target, prediction, mask)
        ensemble_replicate_rows.append(
            _row_metrics(identity=row_identity, metrics=ensemble_metrics)
        )
        for seed in SEEDS:
            member_replicate_rows.append(
                _row_metrics(
                    identity={**row_identity, "seed": seed},
                    metrics=batch.member_metrics[seed],
                )
            )
        mask_rows.append(
            {
                **row_identity,
                "mask_seed": batch.mask_seed,
                "expected_mask_checksum": batch.expected_mask_checksum,
                "regenerated_mask_checksum": batch.regenerated_mask_checksum,
                "mask_array_sha256": mask_sha,
                "target_array_sha256": target_sha,
                "ordered_gene_sha256": batch.ordered_gene_sha256,
                "n_masked": int(np.count_nonzero(mask)),
                "mask_density": float(np.mean(mask)),
                "ensemble_rule": batch.ensemble_rule,
                "ensemble_rule_verified": batch.ensemble_rule_verified,
                "prediction_scale": batch.prediction_scale,
                "graph_null_rule": (
                    GRAPH_NULL_RULE
                    if batch.graph_condition == "node_label_permuted"
                    else None
                ),
                "graph_null_verified": batch.graph_null_verified,
                "degree_sequence_preserved": batch.degree_sequence_preserved,
                "topology_preserved": batch.topology_preserved,
                "permutation_seed": batch.permutation_seed,
            }
        )

        if batch.model_key == GENEMAE and batch.graph_condition == "observed":
            zero = np.zeros_like(target)
            gene_mean = np.broadcast_to(
                np.mean(target, axis=0, dtype=np.float64).astype(
                    target.dtype, copy=False
                ),
                target.shape,
            )
            for reference_key, reference_prediction in (
                ("all_zero", zero),
                ("per_core_gene_mean", gene_mean),
            ):
                baseline_replicate_rows.append(
                    _row_metrics(
                        identity={
                            "model_key": reference_key,
                            "core_alias": batch.core_alias,
                            "mask_rate": float(batch.mask_rate),
                            "replicate": batch.replicate,
                            "graph_condition": "observed",
                        },
                        metrics=masked_regression_metrics(
                            target, reference_prediction, mask
                        ),
                    )
                )

    expected = _expected_batch_keys()
    if seen != expected:
        missing = sorted(expected - seen, key=str)
        extra = sorted(seen - expected, key=str)
        raise GeneMAEComparisonError(
            f"evaluation batch coverage is incomplete; missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )

    all_ensemble_rows = ensemble_replicate_rows + baseline_replicate_rows
    ensemble_core_rows = collapse_mask_replicates(
        all_ensemble_rows,
        identity_fields=("model_key", "core_alias", "mask_rate", "graph_condition"),
    )
    member_core_rows = collapse_mask_replicates(
        member_replicate_rows,
        identity_fields=(
            "model_key",
            "seed",
            "core_alias",
            "mask_rate",
            "graph_condition",
        ),
    )
    ensemble_equal_core_rows = equal_core_aggregates(
        ensemble_core_rows,
        identity_fields=("model_key", "mask_rate", "graph_condition"),
    )
    member_equal_core_rows = equal_core_aggregates(
        member_core_rows,
        identity_fields=("model_key", "seed", "mask_rate", "graph_condition"),
    )
    gates, comparison_rows = evaluate_frozen_gates(
        core_rows=ensemble_core_rows,
        equal_core_rows=ensemble_equal_core_rows,
        member_equal_core_rows=member_equal_core_rows,
    )
    gate_status = {str(row["gate"]): bool(row["passed"]) for row in gates}
    failed_gates = [name for name, passed in gate_status.items() if not passed]
    primary_passed = gate_status["primary_comparison"]
    outcome = "supported" if primary_passed else "negative"
    maximum_conclusion = _maximum_conclusion(
        primary_passed=primary_passed, failed_gates=failed_gates
    )
    core_effect_distributions = _core_effect_distribution_rows(comparison_rows)
    audit_provenance = _mapping(
        audit.provenance, label="comparison audit provenance"
    )
    current_bagm_context = dict(
        _mapping(
            audit_provenance.get("current_bagm_whole_node_context"),
            label="current BAGM context",
        )
    )
    source_historical_context = dict(
        _mapping(
            audit_provenance.get("source_report_historical_context"),
            label="source historical context",
        )
    )
    control_contrasts = _control_contrast_rows(
        ensemble_equal_core_rows,
        current_bagm_context=current_bagm_context,
    )
    control_negative_results = _control_negative_results(control_contrasts)
    analysis = {
        "schema_version": 1,
        "artifact_kind": "myjju_genemae_10core_comparison",
        "campaign_id": CAMPAIGN_ID,
        "current_bagm_campaign_id": CURRENT_BAGM_CAMPAIGN_ID,
        "status": "complete",
        "outcome": outcome,
        "exploratory": True,
        "scientific_question": (
            "Does the reproduced seven-member GeneMAE ensemble improve held-in "
            "20% partial-gene reconstruction over the current seven-member BAGM "
            "GAT ensemble on an identical normalized-log target and masks?"
        ),
        "estimand": "held_in_partial_gene_reconstruction",
        "primary_mask_rate": COMMON_MASK_RATE,
        "native_genemae_mask_rate": NATIVE_MASK_RATE,
        "target_scale": TARGET_SCALE,
        "aggregation": "mask_replicates_then_equal_core",
        "biological_unit": "tissue_core",
        "core_count": len(ALIASES),
        "model_seeds": list(SEEDS),
        "seeds_are_biological_replicates": False,
        "total_fit_cells": 117_386,
        "prediction_ensembles": {
            GENEMAE: GENEMAE_ENSEMBLE_RULE,
            BAGM_GAT: BAGM_ENSEMBLE_RULE,
            BAGM_SELF: BAGM_ENSEMBLE_RULE,
            "metrics_recomputed_after_prediction_combination": True,
            "mean_member_metrics_used_as_ensemble": False,
        },
        "common_scale_caveat": (
            "BAGM decoded counts use the true full-cell library size for CP10k "
            "rescaling; this is an oracle-scale descriptive comparison."
        ),
        "comparison_scope": (
            "end_to_end_trained_system_comparison_not_controlled_architecture_ablation"
        ),
        "system_differences": [
            "training_objective",
            "training_masking_scheme",
            "normalization_and_target_scale",
            "permitted_covariates",
            "graph_construction",
        ],
        "per_core_gene_mean_reference": {
            "target_derived": True,
            "uses_all_fit_cells_and_hidden_targets": True,
            "role": "oracle_descriptive_reference",
            "gene_pearson_interpretation": (
                "undefined_or_numerically_degenerate_because_each_gene_"
                "prediction_is_constant_within_core"
            ),
        },
        "current_bagm_whole_node_context": current_bagm_context,
        "source_report_historical_context": source_historical_context,
        "graph_null": GRAPH_NULL_RULE,
        "frozen_gates": gates,
        "failed_gates": failed_gates,
        "coverage": {
            "expected_batches": len(expected),
            "observed_batches": len(seen),
            "genemae_members": len(audit.members[GENEMAE]),
            "bagm_gat_members": len(audit.members[BAGM_GAT]),
            "bagm_self_members": len(audit.members[BAGM_SELF]),
            "registered_failures_before_completion": len(
                audit.failure_inventory
            ),
            "registered_pilot_attempts": len(audit.pilot_inventory),
            "registered_genemae_pilot_attempts": sum(
                1
                for row in audit.pilot_inventory
                if row.get("model_key") == GENEMAE
            ),
            "registered_genemae_pilot_failures": sum(
                1
                for row in audit.failure_inventory
                if row.get("model_key") == GENEMAE
                and row.get("stage") == "pilot"
            ),
        },
        "negative_results": [
            f"{name} failed its frozen threshold" for name in failed_gates
        ]
        + control_negative_results
        + [
            (
                "The per-core gene-mean predictor is constant within each "
                "core for a given gene, so its gene Pearson is undefined or "
                "numerically degenerate and is not interpretable as evidence "
                "of absent association."
            )
        ],
        "control_contrast_table": "control_contrasts",
        "maximum_defensible_conclusion": maximum_conclusion,
        "uncertainty": {
            "population_confidence_interval": None,
            "bootstrap_used": False,
            "reason": (
                "all cores are held in and coupled by shared fitted weights; "
                "cells, masks, and model seeds are not independent biological "
                "replicates"
            ),
            "descriptive_core_effect_table": "core_effect_distributions",
        },
        "limitations": [
            "all ten cores and all cells were used for fitting",
            "held-in transductive reconstruction is not generalization",
            "partial-gene masking can be dominated by same-cell co-expression",
            "full-cell CP10k normalization uses hidden entries in its denominator",
            "BAGM common-scale conversion uses the true full-cell library size",
            (
                "the per-core gene-mean control is target-derived and uses all "
                "fit cells, including hidden target entries"
            ),
            (
                "this compares end-to-end trained systems that differ in "
                "objective, masking, normalization, covariates, and graph; it "
                "does not isolate an architecture effect"
            ),
            "adjacent-normal tissue is not true Normal",
            "cores coupled by shared fitted weights are not independent model fits",
            (
                "no population confidence interval is reported because the "
                "held-in coupled cores do not support population inference"
            ),
            "model seeds and mask replicates are not biological replicates",
            "the fixed graph null tests graph use but not a biological mechanism",
            "the targeted panel limits biological interpretation",
        ],
        "prohibited_claims": [
            "evaluation of missing historical MyJJu weights",
            "patient-held-out or core-held-out generalization",
            "true-Normal performance",
            "whole-node performance from GeneMAE",
            "direct cellular interaction",
            "biological mechanism",
            "causality",
        ],
        "source_native_50_percent_results_are_ranked_against_bagm": False,
        "protected_identifiers_emitted": False,
    }
    tables = {
        "run_audit": _run_audit_rows(audit),
        "attempt_inventory": [dict(row) for row in audit.attempt_inventory],
        "registered_failures": [dict(row) for row in audit.failure_inventory],
        "pilot_inventory": [dict(row) for row in audit.pilot_inventory],
        "run_resources_and_convergence": _resource_rows(audit),
        "mask_and_target_audit": mask_rows,
        "ensemble_replicate_metrics": ensemble_replicate_rows,
        "baseline_replicate_metrics": baseline_replicate_rows,
        "member_replicate_metrics": member_replicate_rows,
        "ensemble_core_metrics": ensemble_core_rows,
        "member_core_metrics": member_core_rows,
        "ensemble_equal_core_metrics": ensemble_equal_core_rows,
        "member_equal_core_metrics": member_equal_core_rows,
        "seed_variability": _seed_variability_rows(member_equal_core_rows),
        "core_and_seed_comparisons": comparison_rows,
        "core_effect_distributions": core_effect_distributions,
        "control_contrasts": control_contrasts,
        "gate_results": gates,
    }
    provenance = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "run_ids": {
            model_key: [
                member.run_id
                for member in sorted(
                    audit.members[model_key], key=lambda item: item.seed
                )
            ]
            for model_key in MODEL_KEYS
        },
        "checkpoint_sha256": {
            f"{model_key}:seed-{member.seed}": member.checkpoint_sha256
            for model_key in MODEL_KEYS
            for member in audit.members[model_key]
        },
        "provider": dict(provider.provenance()),
        "audit": dict(audit.provenance or {}),
        "read_only_registry_audit": True,
        "protected_identifiers_emitted": False,
    }
    _assert_alias_safe_payload(analysis, label="analysis")
    _assert_alias_safe_payload(tables, label="tables")
    _assert_alias_safe_payload(provenance, label="provenance")
    return analysis, tables, provenance


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_report(
    analysis: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    equal_rows = _indexed(
        tables["ensemble_equal_core_metrics"],
        ("model_key", "mask_rate", "graph_condition"),
    )
    lines = [
        "# MyJJu GeneMAE versus current BAGM",
        "",
        "## Outcome",
        "",
        str(analysis["maximum_defensible_conclusion"]),
        "",
        (
            "This is exploratory held-in partial-gene reconstruction on ten "
            "pathology-confirmed adjacent-normal core aliases. It is not a "
            "generalization, interaction, mechanism, or causal result."
        ),
        "",
        "## Common 20% task",
        "",
        "| Model | Graph | Huber ↓ | MAE ↓ | Pooled Pearson ↑ | Mean gene Pearson ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model_key, condition in (
        (GENEMAE, "observed"),
        (GENEMAE, "node_label_permuted"),
        (BAGM_GAT, "observed"),
        (BAGM_SELF, "observed"),
        ("per_core_gene_mean", "observed"),
        ("all_zero", "observed"),
    ):
        row = equal_rows[(model_key, COMMON_MASK_RATE, condition)]
        lines.append(
            "| "
            + " | ".join(
                (
                    model_key,
                    condition,
                    _format_metric(row.get("masked_huber")),
                    _format_metric(row.get("masked_mae")),
                    _format_metric(row.get("pooled_pearson")),
                    _format_metric(row.get("gene_pearson_mean")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "BAGM probabilities and continuous heads were combined before "
            "decoding; its decoded counts were then transformed with the true "
            "full-cell library size. The latter makes the common scale oracle "
            "and descriptive.",
            (
                "This is an end-to-end trained-system comparison, not a "
                "controlled architecture ablation. GeneMAE and BAGM differ in "
                "training objective, masking, normalization, permitted "
                "covariates, and graph construction."
            ),
            (
                "The per-core gene-mean control is a target-derived oracle: it "
                "uses all fit cells, including the hidden target entries."
            ),
            "",
            "## Control contrasts and metric dependence",
            "",
        ]
    )
    control_findings = _control_negative_results(
        tables["control_contrasts"]
    )
    lines.extend(
        control_findings
        or [
            (
                "The prespecified all-zero and matched-self control contrasts "
                "did not add a directional negative finding."
            )
        ]
    )
    lines.extend(
        [
            (
                "The per-core gene-mean predictor is constant within each core "
                "for a given gene. Its mean gene Pearson is therefore undefined "
                "or numerically degenerate and should not be interpreted."
            ),
            "",
            "## Frozen gates",
            "",
            "| Gate | Result | Observed | Frozen threshold |",
            "|---|---|---|---|",
        ]
    )
    for gate in analysis["frozen_gates"]:
        observed = ", ".join(
            f"{key}={_format_metric(value)}"
            for key, value in gate["observed"].items()
        )
        threshold = ", ".join(
            f"{key}={_format_metric(value)}"
            for key, value in gate["threshold"].items()
        )
        lines.append(
            f"| {gate['gate']} | {'PASS' if gate['passed'] else 'FAIL'} | "
            f"{observed} | {threshold} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive across-core effects",
            "",
            (
                "Positive relative Huber improvement means the candidate has "
                "strictly lower loss. These are descriptive distributions only; "
                "no population confidence interval or bootstrap is reported "
                "because all cores are held in and coupled by shared fitted "
                "weights."
            ),
            "",
            "| Contrast | Cores | Median | Minimum | Maximum | Favoring cores |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tables["core_effect_distributions"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["contrast"]),
                    _format_metric(row["n_cores"]),
                    _format_metric(row["median"]),
                    _format_metric(row["minimum"]),
                    _format_metric(row["maximum"]),
                    _format_metric(row["favoring_core_count"]),
                )
            )
            + " |"
        )
    current = _mapping(
        analysis["current_bagm_whole_node_context"],
        label="current BAGM report context",
    )
    native = equal_rows[(GENEMAE, NATIVE_MASK_RATE, "observed")]
    native_null = equal_rows[
        (GENEMAE, NATIVE_MASK_RATE, "node_label_permuted")
    ]
    lines.extend(
        [
            "",
            "## Current BAGM whole-node result (separate context)",
            "",
            (
                "The immutable current BAGM campaign reported whole-node hybrid "
                f"loss {_format_metric(current['gat_equal_core_hybrid_loss'], 6)} "
                "for the GAT ensemble versus "
                f"{_format_metric(current['matched_self_equal_core_hybrid_loss'], 6)} "
                "for matched self: "
                f"{100.0 * float(current['gat_relative_graph_improvement']):.2f}% "
                "graph gain, favoring GAT in "
                f"{current['gat_favoring_core_count']}/10 cores and "
                f"{current['gat_favoring_seed_pair_count']}/7 seed pairs."
            ),
            (
                "That graph gate passed, but the current campaign outcome was "
                "negative because its pooled-data and representation gates "
                "failed. Whole-node hybrid loss is a different estimand and is "
                "not ranked against GeneMAE partial-gene metrics."
            ),
            "",
            "## Source-native 50% GeneMAE task",
            "",
            f"- Observed-graph Huber: {_format_metric(native['masked_huber'])}",
            f"- Permuted-graph Huber: {_format_metric(native_null['masked_huber'])}",
            "- These 50% results are not ranked against BAGM.",
            "",
            "## Historical source report (non-comparable context)",
            "",
        ]
    )
    source = _mapping(
        analysis["source_report_historical_context"],
        label="source historical report context",
    )
    source_metrics = _mapping(
        source["metrics"], label="source historical report metrics"
    )
    lines.extend(
        [
            (
                "- SO1 sb50: Pearson "
                f"{_format_metric(source_metrics['so1_sb50']['pooled_pearson'])}, "
                "Spearman "
                f"{_format_metric(source_metrics['so1_sb50']['spearman'])}, "
                f"R2 {_format_metric(source_metrics['so1_sb50']['r2'])}, "
                f"MSE {_format_metric(source_metrics['so1_sb50']['mse'])}, "
                "mean per-gene Pearson "
                f"{_format_metric(source_metrics['so1_sb50']['mean_per_gene_pearson'])}, "
                "shuffled-feature Pearson "
                f"{_format_metric(source_metrics['so1_sb50']['shuffled_feature_pearson'])}."
            ),
            (
                "- SO2 sb50: Pearson "
                f"{_format_metric(source_metrics['so2_sb50']['pooled_pearson'])}, "
                "Spearman "
                f"{_format_metric(source_metrics['so2_sb50']['spearman'])}, "
                f"R2 {_format_metric(source_metrics['so2_sb50']['r2'])}, "
                f"MSE {_format_metric(source_metrics['so2_sb50']['mse'])}, "
                "mean per-gene Pearson "
                f"{_format_metric(source_metrics['so2_sb50']['mean_per_gene_pearson'])}, "
                "shuffled-feature Pearson "
                f"{_format_metric(source_metrics['so2_sb50']['shuffled_feature_pearson'])}."
            ),
            (
                "No historical weights are available. These source-reported "
                "values used held-out data for checkpoint selection/scoring and "
                "are checksum-bound context only, not an independent or "
                "common-task comparison."
            ),
            (
                "The source constructor was minimally repaired by assigning "
                "`self.self_hidden` from its constructor argument, then seven "
                "models were retrained from scratch. This report does not "
                "evaluate the missing original weights."
            ),
            "",
            "## Variability, failures, and resources",
            "",
            (
                f"All 7 GeneMAE, 7 BAGM GAT, and 7 BAGM matched-self final "
                f"checkpoints were audited. Registered failed attempts retained "
                f"in the report: {analysis['coverage']['registered_failures_before_completion']}."
            ),
            (
                "Complete per-core, per-seed, convergence, runtime, peak VRAM, "
                "peak host-memory, mask, checksum, and failure tables accompany "
                "this report."
            ),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in analysis["limitations"])
    lines.extend(
        [
            "",
            "## Maximum claim",
            "",
            str(analysis["maximum_defensible_conclusion"]),
            "",
        ]
    )
    return "\n".join(lines)


def html_report(
    analysis: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    markdown = markdown_report(analysis, tables)
    sections: list[str] = []
    in_list = False
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("# "):
            sections.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_list:
                sections.append("</ul>")
                in_list = False
            if in_table:
                sections.append("</tbody></table>")
                in_table = False
            sections.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                sections.append("<ul>")
                in_list = True
            sections.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            if not in_table:
                sections.append("<table><tbody>")
                in_table = True
            tag = "th" if not any(part.startswith("<tr>") for part in sections[-1:]) else "td"
            sections.append(
                "<tr>"
                + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells)
                + "</tr>"
            )
        elif line:
            if in_list:
                sections.append("</ul>")
                in_list = False
            if in_table:
                sections.append("</tbody></table>")
                in_table = False
            sections.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        sections.append("</ul>")
    if in_table:
        sections.append("</tbody></table>")
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MyJJu GeneMAE versus current BAGM</title>
<style>
:root { color-scheme: light; --ink:#17202a; --muted:#5d6d7e; --line:#c8d1da; --panel:#f5f7f9; }
body { max-width:1100px; margin:0 auto; padding:2rem; color:var(--ink); background:white;
       font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
h1,h2 { line-height:1.2; } h2 { margin-top:2rem; border-bottom:1px solid var(--line); padding-bottom:.35rem; }
table { width:100%; border-collapse:collapse; margin:1rem 0; font-variant-numeric:tabular-nums; }
th,td { border:1px solid var(--line); padding:.45rem .55rem; text-align:left; vertical-align:top; }
th { background:var(--panel); } p,li { max-width:90ch; }
@media print { body { max-width:none; padding:0; } h2 { break-after:avoid; } table { break-inside:avoid; } }
</style>
</head>
<body>
""" + "\n".join(sections) + """
</body>
</html>
"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GeneMAEComparisonError("output contains NaN or infinity")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise GeneMAEComparisonError(
        f"unsupported report value {type(value).__name__}"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted(set().union(*(set(row) for row in rows))) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {
                field: (
                    json.dumps(_jsonable(value), sort_keys=True, allow_nan=False)
                    if isinstance(value, (Mapping, list, tuple))
                    else _jsonable(value)
                )
                for field, value in row.items()
            }
            writer.writerow(encoded)


def _assert_portable_html(document: str) -> None:
    lowered = document.lower()
    if (
        "<!doctype html>" not in lowered
        or "<html" not in lowered
        or "<style>" not in lowered
        or "<script" in lowered
        or "src=\"http" in lowered
        or "href=\"http" in lowered
        or "file://" in lowered
    ):
        raise GeneMAEComparisonError("HTML report is not portable and self-contained")


def publish_report(
    *,
    output_dir: Path,
    analysis: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish Markdown, portable HTML, CSV/JSON, and checksums."""

    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise GeneMAEComparisonError(
            f"analysis output already exists and will not be overwritten: {output_dir}"
        )
    _assert_alias_safe_payload(analysis, label="analysis")
    _assert_alias_safe_payload(tables, label="tables")
    _assert_alias_safe_payload(provenance, label="provenance")
    markdown = markdown_report(analysis, tables)
    document = html_report(analysis, tables)
    _assert_portable_html(document)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_json(temporary / "comparison.json", analysis)
        _write_json(temporary / "provenance.json", provenance)
        for name, rows in sorted(tables.items()):
            _write_json(temporary / f"{name}.json", list(rows))
            _write_csv(temporary / f"{name}.csv", list(rows))
        (temporary / "report.md").write_text(markdown, encoding="utf-8")
        (temporary / "report.html").write_text(document, encoding="utf-8")
        files = {
            path.relative_to(temporary).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "artifact_kind": "myjju_genemae_10core_comparison",
            "campaign_id": CAMPAIGN_ID,
            "portable_single_file_html": True,
            "protected_identifiers_emitted": False,
            "files": files,
            "comparison_sha256": files["comparison.json"]["sha256"],
            "provenance_sha256": files["provenance.json"]["sha256"],
        }
        _write_json(temporary / "manifest.json", manifest)
        manifest_sha = sha256_file(temporary / "manifest.json")
        os.rename(temporary, output_dir)
        return {
            **manifest,
            "manifest_sha256": manifest_sha,
            "output_dir": output_dir.as_posix(),
        }
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def run_comparison(
    *, provider: ComparisonProvider, output_dir: Path
) -> dict[str, Any]:
    """Run the complete comparison and atomically publish its report."""

    analysis, tables, provenance = analyze_provider(provider)
    manifest = publish_report(
        output_dir=output_dir,
        analysis=analysis,
        tables=tables,
        provenance=provenance,
    )
    return {"analysis": analysis, "manifest": manifest}


__all__ = [
    "ALIASES",
    "BAGM_ENSEMBLE_RULE",
    "BAGM_GAT",
    "BAGM_SELF",
    "CAMPAIGN_ID",
    "COMMON_MASK_RATE",
    "ComparisonAudit",
    "ComparisonProvider",
    "DEFAULT_OUTPUT_RELATIVE",
    "EXPECTED_GENE_COUNT",
    "EvaluationBatch",
    "FROZEN_CONTRACT_SHA256",
    "GENEMAE",
    "GENEMAE_ENSEMBLE_RULE",
    "GRAPH_NULL_RULE",
    "GeneMAEComparisonError",
    "MODEL_KEYS",
    "NATIVE_MASK_RATE",
    "REPLICATES",
    "RegisteredRunEvidence",
    "SEEDS",
    "TARGET_SCALE",
    "analyze_provider",
    "array_sha256",
    "collapse_mask_replicates",
    "discover_registered_genemae_production",
    "equal_core_aggregates",
    "evaluate_frozen_gates",
    "html_report",
    "markdown_report",
    "masked_regression_metrics",
    "publish_report",
    "relative_improvement",
    "run_comparison",
    "sha256_file",
    "validate_comparison_audit",
]

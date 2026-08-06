#!/usr/bin/env python3
"""Run post-hoc relaxed categorical-sensitivity shards for tokenized G2.

The prespecified 95-percent accuracy gate failed.  This workflow therefore
labels every output as post-hoc exploratory and cannot alter the completed
campaign conclusion.  GPU work is split into independently restartable,
one-whole-node-mask artifacts:

* ``trained-shard`` evaluates all six trained models once per common probe and
  derives all primary and within-width pairs without recomputation.
* ``identical-shard`` independently loads current-width seed 0 twice.
* ``random-shard`` separately instantiates one current and one wider model for
  one paired control seed in 9100..9107.

``pilot`` is mandatory before full shards.  ``aggregate`` is CPU-only and
requires the exact complete 3 + 3 + 24 shard inventory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch


_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from spatial_benchmark.categorical_sensitivity import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    PROBES_PER_MASK,
    PROTOCOL_BASE_SEED,
    PROTOCOL_VERSION,
    CategoricalSensitivityError,
    PairEstimate,
    ProbeSufficientStatistics,
    estimate_pair,
    evaluate_operational_match,
    locked_protocol_record,
    locked_seed,
    make_rademacher_probe,
    multi_tangent_vjp_statistics,
    observed_token_projection_weights,
    paired_tangent_vjp_statistics,
    preactivation_probe_vjp,
    select_locked_whole_node_masks,
    whole_node_targets,
)
from spatial_benchmark.configuration import load_yaml_mapping  # noqa: E402
from spatial_benchmark.expression_tokens import (  # noqa: E402
    audit_expression_tokens,
    tokenize_expression_counts,
)
from spatial_benchmark.full_core import (  # noqa: E402
    FullCoreData,
    ReceiverSortedGraph,
    build_exact_mutual_knn_graph,
    load_and_refit_full_core,
)
from spatial_benchmark.masking import (  # noqa: E402
    FixedMaskBundle,
    create_fixed_mask_bundle,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunValidationError,
    verify_run_bundle,
)
from spatial_benchmark.training import set_deterministic_seed  # noqa: E402
from spatial_benchmark.tokenized_g2 import (  # noqa: E402
    TokenizedReceiverChunkedEdgeConditionedGATv2,
)


_CAMPAIGN_ID = "cmp_20260726_full_core_g2_count_tokens_multiseed"
_CURRENT_VARIANT = "g2_tokenized_width512_exact_k1000_full_core"
_WIDER_VARIANT = "g2_tokenized_width1024_exact_k1000_full_core"
_TASK = "masked_expression_token_classification"
_MODEL_CLASS = (
    "spatial_benchmark.tokenized_g2."
    "TokenizedReceiverChunkedEdgeConditionedGATv2"
)
_ANALYSIS_MODE = "post_hoc_exploratory_after_prespecified_gate_failure"
_ANALYSIS_SCHEMA_VERSION = 1
_SHARD_SCHEMA_VERSION = 1
_ACTIVE_ANALYSIS_LAYOUT_VERSION = 1
_NODE_PROJECTION_CHUNK_SIZE = 256
_RANDOM_SEEDS = tuple(range(9100, 9108))
_TRAINED_PAIRS = (
    ("current_s0", "wider_s0"),
    ("current_s1", "wider_s1"),
    ("current_s2", "wider_s2"),
    ("current_s0", "current_s1"),
    ("current_s0", "current_s2"),
    ("current_s1", "current_s2"),
    ("wider_s0", "wider_s1"),
    ("wider_s0", "wider_s2"),
    ("wider_s1", "wider_s2"),
)
_IDENTICAL_PAIR = ("identical_a", "identical_b")


class SensitivityWorkflowError(CategoricalSensitivityError):
    """Raised when the post-hoc shard workflow violates its contract."""


@dataclass
class RunSpec:
    """Strictly verified model/run identity used by one shard."""

    root: Path
    run_id: str
    variant_label: str
    width_label: str
    seed: int
    config: Mapping[str, Any]
    constructor_arguments: Mapping[str, Any]
    state_dict: Mapping[str, torch.Tensor] | None
    state_dict_sha256: str
    checkpoint_sha256: str
    run_bundle_checksum_manifest_sha256: str
    completion_marker_sha256: str
    config_resolved_sha256: str
    final_metrics_sha256: str
    fixed_mask_provenance_sha256: str
    tokenization_provenance_sha256: str
    graph_sha256: str
    preprocessing_sha256: str
    mask_bundle_sha256: str
    mask_manifest: Mapping[str, Any]
    token_matrix_sha256: str
    exact_accuracy_percent: float
    parameter_count: int
    verified_bundle_file_count: int

    @property
    def label(self) -> str:
        return f"{self.width_label}_s{self.seed}"

    def instantiate_trained(
        self,
    ) -> TokenizedReceiverChunkedEdgeConditionedGATv2:
        if self.state_dict is None:
            raise SensitivityWorkflowError(
                f"checkpoint state for {self.label} was not retained"
            )
        model = TokenizedReceiverChunkedEdgeConditionedGATv2(
            **dict(self.constructor_arguments)
        )
        model.load_state_dict(dict(self.state_dict), strict=True)
        return model

    def instantiate_random(
        self,
        paired_seed: int,
    ) -> TokenizedReceiverChunkedEdgeConditionedGATv2:
        # The locked control separately instantiates both widths after
        # resetting this same paired seed. Aligned shapes can share RNG
        # prefixes; this conservative limitation is reported.
        set_deterministic_seed(int(paired_seed), deterministic=True)
        return TokenizedReceiverChunkedEdgeConditionedGATv2(
            **dict(self.constructor_arguments)
        )

    def provenance(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_path": str(self.root),
            "variant_label": self.variant_label,
            "width_label": self.width_label,
            "seed": self.seed,
            "constructor_arguments": dict(self.constructor_arguments),
            "state_dict_sha256": self.state_dict_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "run_bundle_checksum_manifest_sha256": (
                self.run_bundle_checksum_manifest_sha256
            ),
            "completion_marker_sha256": self.completion_marker_sha256,
            "config_resolved_sha256": self.config_resolved_sha256,
            "final_metrics_sha256": self.final_metrics_sha256,
            "fixed_mask_provenance_sha256": (
                self.fixed_mask_provenance_sha256
            ),
            "tokenization_provenance_sha256": (
                self.tokenization_provenance_sha256
            ),
            "graph_sha256": self.graph_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "mask_bundle_sha256": self.mask_bundle_sha256,
            "token_matrix_sha256": self.token_matrix_sha256,
            "whole_node_exact_accuracy_percent": (
                self.exact_accuracy_percent
            ),
            "parameter_count": self.parameter_count,
            "standard_bundle_verification": {
                "valid": True,
                "status": "success",
                "file_count": self.verified_bundle_file_count,
            },
        }


@dataclass(frozen=True)
class DeviceInputs:
    expression: torch.Tensor
    node_covariates: torch.Tensor
    edge_index: torch.Tensor
    edge_attributes: torch.Tensor


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SensitivityWorkflowError(f"{label} must be a mapping")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SensitivityWorkflowError(f"{label} must be an integer")
    return int(value)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SensitivityWorkflowError(f"{label} must be numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise SensitivityWorkflowError(f"{label} must be numeric") from exc
    if not math.isfinite(converted):
        raise SensitivityWorkflowError(f"{label} must be finite")
    return converted


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SensitivityWorkflowError(
            f"cannot read valid JSON from {path}"
        ) from exc
    return _mapping(value, str(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = torch.as_tensor(
            state_dict[name]
        ).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(_canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".sha256.json")


def _write_bound_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    analysis_input_sha256: str,
) -> None:
    sidecar_path = _sidecar_path(path)
    path_exists = path.exists()
    if path_exists and sidecar_path.exists():
        raise SensitivityWorkflowError(
            f"refusing to overwrite existing artifact {path}"
        )
    if path_exists:
        existing = _load_json(path)
        if (
            existing.get("analysis_input_sha256")
            != analysis_input_sha256
            or _canonical_json(existing) != _canonical_json(payload)
        ):
            raise SensitivityWorkflowError(
                f"incomplete artifact conflicts with recomputation: {path}"
            )
        _atomic_json(
            sidecar_path,
            {
                "schema_version": 1,
                "analysis_input_sha256": analysis_input_sha256,
                "artifact_sha256": _sha256_file(path),
            },
        )
        return
    # A sidecar without its JSON can only be an interrupted publication.
    # The caller holds the artifact lock, so recomputation may safely replace
    # that incomplete sidecar after atomically publishing the new JSON.
    _atomic_json(path, payload)
    _atomic_json(
        sidecar_path,
        {
            "schema_version": 1,
            "analysis_input_sha256": analysis_input_sha256,
            "artifact_sha256": _sha256_file(path),
        },
    )


def _verify_bound_json(
    path: Path,
    *,
    analysis_input_sha256: str,
) -> Mapping[str, Any]:
    sidecar_path = _sidecar_path(path)
    if not path.is_file():
        raise SensitivityWorkflowError(
            f"artifact is missing: {path}"
        )
    payload = _load_json(path)
    if payload.get("analysis_input_sha256") != analysis_input_sha256:
        raise SensitivityWorkflowError(
            f"artifact belongs to different analysis inputs: {path}"
        )
    if not sidecar_path.is_file():
        # The JSON rename is the authoritative atomic publication. Repair a
        # missing derived checksum sidecar after a crash between the two
        # renames; callers hold an analysis/artifact lock.
        _atomic_json(
            sidecar_path,
            {
                "schema_version": 1,
                "analysis_input_sha256": analysis_input_sha256,
                "artifact_sha256": _sha256_file(path),
            },
        )
    sidecar = _load_json(sidecar_path)
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("analysis_input_sha256") != analysis_input_sha256
        or sidecar.get("artifact_sha256") != _sha256_file(path)
    ):
        raise SensitivityWorkflowError(
            f"artifact checksum binding failed: {path}"
        )
    return payload


class _ExclusiveLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: io.TextIOWrapper | None = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, *args: object) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise SensitivityWorkflowError(
            f"cannot load checkpoint {path}"
        ) from exc
    return _mapping(payload, str(path))


def _load_run(
    root: str | Path,
    *,
    expected_variant: str,
    expected_seed: int,
    retain_state: bool,
) -> RunSpec:
    path = Path(root).resolve()
    run_id = path.name
    try:
        bundle = verify_run_bundle(path)
    except RunValidationError as exc:
        raise SensitivityWorkflowError(
            f"{path} is not a checksum-valid finalized bundle"
        ) from exc
    if bundle.get("status") != "success":
        raise SensitivityWorkflowError(f"{path} is not successful")

    config = load_yaml_mapping(path / "config.resolved.yaml")
    campaign = _mapping(config.get("campaign"), f"{run_id} campaign")
    experiment = _mapping(config.get("experiment"), f"{run_id} experiment")
    evaluation = _mapping(config.get("evaluation"), f"{run_id} evaluation")
    trainer = _mapping(config.get("trainer"), f"{run_id} trainer")
    if campaign.get("campaign_id") != _CAMPAIGN_ID:
        raise SensitivityWorkflowError(f"{run_id} has the wrong campaign")
    if experiment.get("variant_label") != expected_variant:
        raise SensitivityWorkflowError(f"{run_id} has the wrong variant")
    if _integer(config.get("seed"), f"{run_id} seed") != expected_seed:
        raise SensitivityWorkflowError(f"{run_id} has the wrong seed")
    if (
        evaluation.get("task_family") != _TASK
        or _integer(
            evaluation.get("mask_replicates_per_mode"),
            f"{run_id} mask replicates",
        )
        != 3
        or _integer(trainer.get("max_epochs"), f"{run_id} epochs") != 200
    ):
        raise SensitivityWorkflowError(
            f"{run_id} training/evaluation contract drifted"
        )

    checkpoint_path = path / "checkpoints" / "last.ckpt"
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("run_id") != run_id
        or checkpoint.get("task_family") != _TASK
        or checkpoint.get("checkpoint_role") != "last"
        or checkpoint.get("checkpoint_policy")
        != "final_epoch_no_validation_selection"
        or _integer(checkpoint.get("epoch"), f"{run_id} checkpoint epoch")
        != 199
        or _integer(
            checkpoint.get("fixed_epoch_budget"),
            f"{run_id} checkpoint budget",
        )
        != 200
    ):
        raise SensitivityWorkflowError(
            f"{run_id} checkpoint semantics drifted"
        )
    construction = _mapping(
        checkpoint.get("model_construction"),
        f"{run_id} model construction",
    )
    if (
        construction.get("canonical_model_key") != "g2tokenized"
        or construction.get("implementation_class") != _MODEL_CLASS
    ):
        raise SensitivityWorkflowError(
            f"{run_id} has the wrong model implementation"
        )
    arguments = dict(
        _mapping(
            construction.get("constructor_arguments"),
            f"{run_id} constructor arguments",
        )
    )
    expected_hidden = 512 if expected_variant == _CURRENT_VARIANT else 1024
    expected_heads = 4 if expected_variant == _CURRENT_VARIANT else 8
    expected_arguments = {
        "num_genes": 1000,
        "node_covariate_dim": 22,
        "edge_attribute_dim": 17,
        "hidden_dim": expected_hidden,
        "attention_heads": expected_heads,
        "graph_layers": 2,
        "ffn_dim": expected_hidden,
        "decoder_dim": expected_hidden,
        "edge_hidden_dim": 64,
        "edge_embedding_dim": 64,
        "num_expression_tokens": 4,
        "activation_checkpointing": True,
    }
    if any(arguments.get(key) != value for key, value in expected_arguments.items()):
        raise SensitivityWorkflowError(
            f"{run_id} constructor dimensions drifted"
        )
    if (
        arguments.get("attention_head_dim") is not None
        or float(arguments.get("dropout", float("nan"))) != 0.1
        or float(arguments.get("attention_dropout", float("nan"))) != 0.1
        or int(arguments.get("receiver_chunk_size", 0)) <= 0
    ):
        raise SensitivityWorkflowError(
            f"{run_id} constructor execution contract drifted"
        )
    raw_state = _mapping(
        checkpoint.get("model_state_dict"),
        f"{run_id} state dict",
    )
    if not all(
        isinstance(name, str) and torch.is_tensor(value)
        for name, value in raw_state.items()
    ):
        raise SensitivityWorkflowError(
            f"{run_id} state dict is malformed"
        )
    observed_state_sha = _state_dict_sha256(raw_state)
    recorded_state_sha = str(checkpoint.get("state_dict_sha256", ""))
    if observed_state_sha != recorded_state_sha:
        raise SensitivityWorkflowError(
            f"{run_id} state checksum mismatch"
        )

    mask_record = _load_json(
        path / "provenance" / "fixed_evaluation_masks.json"
    )
    mask_manifest = _mapping(
        mask_record.get("bundle_manifest"),
        f"{run_id} mask manifest",
    )
    if (
        mask_record.get("used_for_gradient_updates") is not False
        or mask_record.get("used_for_checkpoint_selection") is not False
        or checkpoint.get("evaluation_mask_bundle_sha256")
        != mask_manifest.get("bundle_checksum")
    ):
        raise SensitivityWorkflowError(
            f"{run_id} fixed-mask provenance is invalid"
        )
    token_record = _load_json(
        path / "diagnostics" / "expression_tokenization.json"
    )
    token_sha = str(token_record.get("token_checksum_sha256", ""))
    if len(token_sha) != 64:
        raise SensitivityWorkflowError(
            f"{run_id} token checksum is missing"
        )
    final_metrics = _load_json(path / "metrics" / "final.json")
    exact_accuracy = _finite(
        final_metrics.get(
            "fit/whole_node/masked_token_accuracy_percent"
        ),
        f"{run_id} exact accuracy",
    )

    return RunSpec(
        root=path,
        run_id=run_id,
        variant_label=expected_variant,
        width_label=(
            "current" if expected_variant == _CURRENT_VARIANT else "wider"
        ),
        seed=expected_seed,
        config=config,
        constructor_arguments=arguments,
        state_dict=dict(raw_state) if retain_state else None,
        state_dict_sha256=recorded_state_sha,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        run_bundle_checksum_manifest_sha256=_sha256_file(
            path / "provenance" / "artifact_checksums.json"
        ),
        completion_marker_sha256=_sha256_file(path / "_SUCCESS"),
        config_resolved_sha256=_sha256_file(path / "config.resolved.yaml"),
        final_metrics_sha256=_sha256_file(path / "metrics" / "final.json"),
        fixed_mask_provenance_sha256=_sha256_file(
            path / "provenance" / "fixed_evaluation_masks.json"
        ),
        tokenization_provenance_sha256=_sha256_file(
            path / "diagnostics" / "expression_tokenization.json"
        ),
        graph_sha256=str(checkpoint.get("graph_sha256", "")),
        preprocessing_sha256=str(
            checkpoint.get("full_core_preprocessing_sha256", "")
        ),
        mask_bundle_sha256=str(
            checkpoint.get("evaluation_mask_bundle_sha256", "")
        ),
        mask_manifest=mask_manifest,
        token_matrix_sha256=token_sha,
        exact_accuracy_percent=exact_accuracy,
        parameter_count=sum(
            int(torch.as_tensor(value).numel())
            for value in raw_state.values()
        ),
        verified_bundle_file_count=int(bundle.get("file_count", 0)),
    )


def _load_runs(
    current_paths: Sequence[str | Path],
    wider_paths: Sequence[str | Path],
    *,
    retain_labels: set[str],
) -> tuple[RunSpec, ...]:
    if len(current_paths) != 3 or len(wider_paths) != 3:
        raise SensitivityWorkflowError(
            "exactly three current and three wider run paths are required"
        )
    runs: list[RunSpec] = []
    for seed in range(3):
        label = f"current_s{seed}"
        runs.append(
            _load_run(
                current_paths[seed],
                expected_variant=_CURRENT_VARIANT,
                expected_seed=seed,
                retain_state=label in retain_labels,
            )
        )
    for seed in range(3):
        label = f"wider_s{seed}"
        runs.append(
            _load_run(
                wider_paths[seed],
                expected_variant=_WIDER_VARIANT,
                expected_seed=seed,
                retain_state=label in retain_labels,
            )
        )
    _validate_common_run_identity(runs)
    return tuple(runs)


def _validate_common_run_identity(runs: Sequence[RunSpec]) -> None:
    if len(runs) != 6 or {run.label for run in runs} != {
        *(f"current_s{seed}" for seed in range(3)),
        *(f"wider_s{seed}" for seed in range(3)),
    }:
        raise SensitivityWorkflowError(
            "run set does not contain the exact six variants/seeds"
        )
    for field in (
        "graph_sha256",
        "preprocessing_sha256",
        "mask_bundle_sha256",
        "token_matrix_sha256",
    ):
        values = {getattr(run, field) for run in runs}
        if len(values) != 1 or len(next(iter(values))) != 64:
            raise SensitivityWorkflowError(
                f"runs disagree on {field}"
            )
    manifests = {
        hashlib.sha256(_canonical_json(run.mask_manifest)).hexdigest()
        for run in runs
    }
    if len(manifests) != 1:
        raise SensitivityWorkflowError(
            "runs disagree on fixed-mask manifest"
        )
    reference = runs[0].config
    for section in ("dataset", "features", "graph", "masking", "evaluation"):
        expected = _canonical_json(
            _mapping(reference.get(section), f"reference {section}")
        )
        for run in runs[1:]:
            if _canonical_json(
                _mapping(run.config.get(section), f"{run.run_id} {section}")
            ) != expected:
                raise SensitivityWorkflowError(
                    f"runs disagree on config section {section}"
                )


def _validate_failed_gate_comparison(
    comparison_path: Path,
    runs: Sequence[RunSpec],
) -> Mapping[str, Any]:
    result = _load_json(comparison_path)
    verified = _mapping(
        result.get("verified_identity"),
        "comparison verified identity",
    )
    if (
        result.get("schema_version") != 1
        or result.get("artifact_kind") != "g2_token_multiseed_comparison"
        or result.get("campaign_id") != _CAMPAIGN_ID
        or result.get("status") != "complete"
        or verified.get("all_six_runs_match") is not True
        or verified.get("fixed_epochs_per_run") != 200
        or verified.get("seeds_per_variant") != [0, 1, 2]
    ):
        raise SensitivityWorkflowError(
            "comparison identity/provenance contract is invalid"
        )
    rows = result.get("runs")
    if not isinstance(rows, Sequence) or len(rows) != 6:
        raise SensitivityWorkflowError(
            "comparison must contain exactly six run rows"
        )
    run_by_key = {
        (run.variant_label, run.seed): run for run in runs
    }
    seen: set[tuple[str, int]] = set()
    for raw_row in rows:
        row = _mapping(raw_row, "comparison run row")
        key = (
            str(row.get("variant_label", "")),
            _integer(row.get("seed"), "comparison run seed"),
        )
        if key in seen or key not in run_by_key:
            raise SensitivityWorkflowError(
                "comparison run rows are duplicated or unexpected"
            )
        seen.add(key)
        run = run_by_key[key]
        if row.get("run_id") != run.run_id or not math.isclose(
            _finite(
                row.get("whole_node_exact_accuracy_percent"),
                "comparison exact accuracy",
            ),
            run.exact_accuracy_percent,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SensitivityWorkflowError(
                "comparison run row disagrees with verified run artifact"
            )
    gate = _mapping(
        result.get("relaxed_jacobian_gate"),
        "comparison relaxed Jacobian gate",
    )
    values = _mapping(
        gate.get("wider_seed_values_percent"),
        "gate wider values",
    )
    for seed in range(3):
        run_value = run_by_key[
            (_WIDER_VARIANT, seed)
        ].exact_accuracy_percent
        gate_value = _finite(values.get(str(seed)), f"gate seed {seed}")
        if not math.isclose(
            run_value,
            gate_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SensitivityWorkflowError(
                "failed gate value disagrees with wider run"
            )
    if (
        gate.get("eligible") is not False
        or gate.get("status") != "skipped"
        or gate.get("jacobians_computed") is not False
        or gate.get("operator") != "strictly_greater_than"
        or _finite(gate.get("threshold_percent"), "gate threshold") != 95.0
        or any(_finite(values[str(seed)], "gate value") > 95.0 for seed in range(3))
    ):
        raise SensitivityWorkflowError(
            "post-hoc workflow requires the recorded failed 95-percent gate"
        )
    return result


def _whole_node_manifest_rows(
    manifest: Mapping[str, Any],
) -> list[dict[str, object]]:
    entries = manifest.get("entries")
    if not isinstance(entries, Sequence):
        raise SensitivityWorkflowError("mask manifest lacks entries")
    selected: list[dict[str, object]] = []
    for raw_entry in entries:
        entry = _mapping(raw_entry, "mask manifest entry")
        spec = _mapping(entry.get("spec"), "mask manifest spec")
        if spec.get("mode") != "node":
            continue
        selected.append(
            {
                "replicate": _integer(
                    entry.get("replicate"),
                    "whole-node replicate",
                ),
                "entry_id": str(entry.get("entry_id", "")),
                "mask_checksum": str(entry.get("mask_checksum", "")),
                "shape": list(entry.get("shape", ())),
                "n_masked": int(
                    _mapping(
                        entry.get("summary"),
                        "whole-node summary",
                    ).get("n_masked_entries", -1)
                ),
            }
        )
    selected.sort(key=lambda row: int(row["replicate"]))
    if (
        [row["replicate"] for row in selected] != [0, 1, 2]
        or any(not row["entry_id"] for row in selected)
        or any(len(str(row["mask_checksum"])) != 64 for row in selected)
    ):
        raise SensitivityWorkflowError(
            "mask manifest lacks exact whole-node replicates 0, 1, and 2"
        )
    return selected


def _analysis_manifest_core(
    *,
    comparison_path: Path,
    comparison: Mapping[str, Any],
    runs: Sequence[RunSpec],
) -> dict[str, object]:
    package_root = _ROOT / "src" / "spatial_benchmark"
    gate = _mapping(
        comparison["relaxed_jacobian_gate"],
        "comparison gate",
    )
    graph_config = _mapping(runs[0].config.get("graph"), "graph config")
    mask_rows = _whole_node_manifest_rows(runs[0].mask_manifest)
    shapes = {tuple(row["shape"]) for row in mask_rows}
    if len(shapes) != 1:
        raise SensitivityWorkflowError(
            "whole-node mask shapes disagree"
        )
    shape = next(iter(shapes))
    if (
        len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        )
    ):
        raise SensitivityWorkflowError(
            "whole-node mask shape is invalid"
        )
    directed_edges = _integer(
        graph_config.get("expected_directed_edges"),
        "expected directed edges",
    )
    if directed_edges <= 0:
        raise SensitivityWorkflowError(
            "expected directed edge count must be positive"
        )
    return {
        "schema_version": _ANALYSIS_SCHEMA_VERSION,
        "artifact_kind": "g2_token_categorical_sensitivity_analysis_inputs",
        "active_analysis_layout_version": _ACTIVE_ANALYSIS_LAYOUT_VERSION,
        "protocol": PROTOCOL_VERSION,
        "analysis_mode": _ANALYSIS_MODE,
        "campaign_id": _CAMPAIGN_ID,
        "prespecified_gate": {
            **dict(gate),
            "post_hoc_override": (
                "user explicitly requested relaxed categorical sensitivity "
                "after the prespecified gate failed"
            ),
        },
        "comparison_path": str(comparison_path),
        "comparison_sha256": _sha256_file(comparison_path),
        "runs": [run.provenance() for run in runs],
        "common_identity": {
            "graph_sha256": runs[0].graph_sha256,
            "preprocessing_sha256": runs[0].preprocessing_sha256,
            "mask_bundle_sha256": runs[0].mask_bundle_sha256,
            "token_matrix_sha256": runs[0].token_matrix_sha256,
            "n_nodes": shape[0],
            "n_genes": shape[1],
            "directed_edges": directed_edges,
            "whole_node_masks": mask_rows,
        },
        "execution_contract": {
            "workflow": "post_hoc_evaluation_of_registered_immutable_runs",
            "scratch_layout": "exclusive_active_analysis_directory",
            "floating_point": "fp32_no_amp_no_tf32",
            "node_projection_chunk_size": _NODE_PROJECTION_CHUNK_SIZE,
            "trained_shard": "one_mask_all_six_models",
            "identical_shard": "one_mask_two_independent_current_s0_loads",
            "identical_fail_fast_review": (
                "CPU review after exactly 3 identical shards and before "
                "trained/random shards"
            ),
            "random_shard": "one_mask_one_paired_seed",
            "expected_full_graph_vjps": 2_304,
            "resource_pilot_required": True,
            "explicit_pilot_review_required": True,
        },
        "source_checksums": {
            "workflow_script_sha256": _sha256_file(Path(__file__).resolve()),
            "spatial_benchmark_package_sha256": {
                path.name: _sha256_file(path)
                for path in sorted(package_root.glob("*.py"))
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "torch_geometric": _package_version("torch-geometric"),
            "torch_scatter": _package_version("torch-scatter"),
        },
        "maximum_claim": (
            "post-hoc local functional sensitivity similarity on one "
            "transductive core; not confirmatory, generalizable, biological, "
            "or causal evidence"
        ),
    }


def _initialize_work_root(
    work_root: Path,
    manifest_core: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    digest = hashlib.sha256(_canonical_json(manifest_core)).hexdigest()
    manifest = {**dict(manifest_core), "analysis_input_sha256": digest}
    with _ExclusiveLock(work_root / "locks" / "initialize.lock"):
        marker_path = work_root / ".g2-token-sensitivity-active-analysis.json"
        path = work_root / "analysis_manifest.json"
        if path.is_file():
            observed = _load_json(path)
            if _canonical_json(observed) != _canonical_json(manifest):
                raise SensitivityWorkflowError(
                    "work root belongs to different analysis inputs"
                )
            marker = _load_json(marker_path)
            if (
                marker.get("artifact_kind")
                != "g2_token_categorical_sensitivity_active_analysis"
                or marker.get("layout_version")
                != _ACTIVE_ANALYSIS_LAYOUT_VERSION
                or marker.get("analysis_input_sha256") != digest
            ):
                raise SensitivityWorkflowError(
                    "active-analysis directory ownership marker is invalid"
                )
        else:
            existing = [
                child.name
                for child in work_root.iterdir()
                if child.name != "locks"
            ] if work_root.is_dir() else []
            if existing:
                raise SensitivityWorkflowError(
                    "new active-analysis directory must be empty"
                )
            work_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(path, manifest)
            _atomic_json(work_root / "protocol.json", locked_protocol_record())
            _atomic_json(
                marker_path,
                {
                    "schema_version": 1,
                    "artifact_kind": (
                        "g2_token_categorical_sensitivity_active_analysis"
                    ),
                    "layout_version": _ACTIVE_ANALYSIS_LAYOUT_VERSION,
                    "analysis_input_sha256": digest,
                    "outputs_are_immutable": True,
                },
            )
    return manifest, digest


def _load_analysis_manifest(work_root: Path) -> Mapping[str, Any]:
    path = work_root / "analysis_manifest.json"
    manifest = _load_json(path)
    if (
        manifest.get("schema_version") != _ANALYSIS_SCHEMA_VERSION
        or manifest.get("artifact_kind")
        != "g2_token_categorical_sensitivity_analysis_inputs"
        or manifest.get("active_analysis_layout_version")
        != _ACTIVE_ANALYSIS_LAYOUT_VERSION
        or manifest.get("protocol") != PROTOCOL_VERSION
        or manifest.get("analysis_mode") != _ANALYSIS_MODE
    ):
        raise SensitivityWorkflowError("analysis manifest is invalid")
    analysis_sha = str(manifest.get("analysis_input_sha256", ""))
    core = dict(manifest)
    core.pop("analysis_input_sha256", None)
    if (
        not _is_sha256(analysis_sha)
        or analysis_sha
        != hashlib.sha256(_canonical_json(core)).hexdigest()
    ):
        raise SensitivityWorkflowError(
            "analysis manifest self-digest is invalid"
        )
    marker = _load_json(
        work_root / ".g2-token-sensitivity-active-analysis.json"
    )
    if (
        marker.get("artifact_kind")
        != "g2_token_categorical_sensitivity_active_analysis"
        or marker.get("layout_version")
        != _ACTIVE_ANALYSIS_LAYOUT_VERSION
        or marker.get("analysis_input_sha256") != analysis_sha
        or marker.get("outputs_are_immutable") is not True
    ):
        raise SensitivityWorkflowError(
            "active-analysis ownership marker is invalid"
        )
    protocol = _load_json(work_root / "protocol.json")
    if _canonical_json(protocol) != _canonical_json(
        locked_protocol_record()
    ):
        raise SensitivityWorkflowError(
            "active-analysis protocol record drifted"
        )
    source = _mapping(manifest.get("source_checksums"), "source checksums")
    if source.get("workflow_script_sha256") != _sha256_file(
        Path(__file__).resolve()
    ):
        raise SensitivityWorkflowError(
            "workflow source changed after analysis initialization"
        )
    package_hashes = _mapping(
        source.get("spatial_benchmark_package_sha256"),
        "package source checksums",
    )
    package_root = _ROOT / "src" / "spatial_benchmark"
    actual_package_hashes = {
        file.name: _sha256_file(file)
        for file in sorted(package_root.glob("*.py"))
    }
    if dict(package_hashes) != actual_package_hashes:
        raise SensitivityWorkflowError(
            "spatial_benchmark source changed after analysis initialization"
        )
    _analysis_run_rows(manifest)
    return manifest


def _require_exact_bound_inventory(
    root: Path,
    expected_artifacts: Sequence[Path],
) -> None:
    expected = {
        path.resolve() for path in expected_artifacts
    } | {
        _sidecar_path(path).resolve() for path in expected_artifacts
    }
    actual = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
    } if root.is_dir() else set()
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise SensitivityWorkflowError(
            "bound shard inventory mismatch; "
            f"missing={missing}, extra={extra}"
        )


def _resolve_prepared_artifact(config: Mapping[str, Any]) -> Path:
    dataset = _mapping(config.get("dataset"), "dataset config")
    raw = dataset.get("prepared_artifact_reference")
    if not isinstance(raw, str) or not raw:
        raise SensitivityWorkflowError(
            "dataset lacks prepared artifact reference"
        )
    path = Path(raw)
    return path if path.is_absolute() else (_ROOT / path).resolve()


def _build_graph(
    core: FullCoreData,
    graph_config: Mapping[str, Any],
) -> ReceiverSortedGraph:
    return build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=int(graph_config.get("k", graph_config.get("neighbor_k", 0))),
        radius_guard_um=float(
            graph_config.get(
                "radius_guard_um",
                graph_config.get("radius_um", 0.0),
            )
        ),
        query_chunk_size=int(graph_config.get("query_chunk_size", 2048)),
        receiver_chunk_size=int(
            graph_config.get("receiver_shard_size", 512)
        ),
        mutual_search_chunk_size=int(
            graph_config.get("mutual_search_chunk_size", 4_000_000)
        ),
        workers=int(graph_config.get("construction_workers", 1)),
        epsilon=float(
            graph_config.get("edge_standardizer_epsilon", 1e-8)
        ),
    )


def _recreate_masks(
    core: FullCoreData,
    manifest: Mapping[str, Any],
) -> FixedMaskBundle:
    entries = manifest.get("entries")
    if not isinstance(entries, Sequence):
        raise SensitivityWorkflowError("mask manifest lacks entries")
    specs: dict[int, Mapping[str, Any]] = {}
    for raw_entry in entries:
        entry = _mapping(raw_entry, "mask entry")
        if _integer(entry.get("replicate"), "mask replicate") == 0:
            specs[
                _integer(entry.get("spec_index"), "mask spec index")
            ] = _mapping(entry.get("spec"), "mask spec")
    if set(specs) != {0, 1, 2}:
        raise SensitivityWorkflowError(
            "mask manifest lacks three specifications"
        )
    bundle = create_fixed_mask_bundle(
        {"fit": core.coordinates_um},
        core.n_genes,
        [specs[index] for index in range(3)],
        replicates=_integer(manifest.get("replicates"), "mask replicates"),
        base_seed=_integer(manifest.get("base_seed"), "mask base seed"),
    )
    if _canonical_json(bundle.manifest) != _canonical_json(manifest):
        raise SensitivityWorkflowError(
            "recreated fixed masks do not match manifest"
        )
    return bundle


def _enforce_cublas_workspace_config() -> None:
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config not in {None, ":4096:8"}:
        raise SensitivityWorkflowError(
            "CUBLAS_WORKSPACE_CONFIG must be unset or :4096:8"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _configure_device(
    name: str,
    *,
    expected_physical_gpu_uuid: str,
) -> tuple[torch.device, Mapping[str, Any]]:
    _enforce_cublas_workspace_config()
    device = torch.device(name)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_tokens = (
        [token.strip() for token in visible.split(",")]
        if visible is not None
        else []
    )
    if (
        device.type != "cuda"
        or device.index != 0
        or len(visible_tokens) != 1
        or not visible_tokens[0]
    ):
        raise SensitivityWorkflowError(
            "GPU work requires process-level isolation with exactly one "
            "CUDA_VISIBLE_DEVICES entry and logical --device cuda:0"
        )
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise SensitivityWorkflowError(
            "isolated CUDA device is unavailable or more than one GPU is visible"
        )
    gpu_identity = _gpu_identity_record(
        device,
        expected_physical_gpu_uuid=expected_physical_gpu_uuid,
    )
    set_deterministic_seed(PROTOCOL_BASE_SEED, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device, gpu_identity


def _gpu_identity_record(
    device: torch.device,
    *,
    expected_physical_gpu_uuid: str,
) -> dict[str, object]:
    if not _is_gpu_uuid(expected_physical_gpu_uuid):
        raise SensitivityWorkflowError(
            "supervisor-assigned physical GPU UUID must use GPU-xxxxxxxx-"
            "xxxx-xxxx-xxxx-xxxxxxxxxxxx form"
        )
    properties = torch.cuda.get_device_properties(device)
    physical_uuid = f"GPU-{properties.uuid}"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SensitivityWorkflowError(
            "cannot resolve physical GPU identity with nvidia-smi"
        ) from exc
    matches: list[tuple[str, str, str, str]] = []
    for line in completed.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) == 4 and fields[1] == physical_uuid:
            matches.append(fields)
    if len(matches) != 1:
        raise SensitivityWorkflowError(
            "logical CUDA device did not map to one physical GPU UUID"
        )
    raw_index, gpu_uuid, pci_bus_id, driver_version = matches[0]
    try:
        physical_index = int(raw_index)
    except ValueError as exc:
        raise SensitivityWorkflowError(
            "nvidia-smi returned a non-integer physical GPU index"
        ) from exc
    if gpu_uuid != expected_physical_gpu_uuid:
        raise SensitivityWorkflowError(
            "logical CUDA device UUID does not match the explicit "
            "supervisor assignment"
        )
    if physical_index in {0, 4}:
        raise SensitivityWorkflowError(
            f"physical GPU {physical_index} is forbidden for this analysis"
        )
    if physical_index not in {5, 6, 7}:
        raise SensitivityWorkflowError(
            "post-hoc shards are restricted to physical GPUs 5, 6, and 7"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible is not None
    if visible.isdigit() and int(visible) != physical_index:
        raise SensitivityWorkflowError(
            "numeric CUDA_VISIBLE_DEVICES disagrees with physical GPU index"
        )
    return {
        "cuda_visible_devices": visible,
        "visible_device_count": torch.cuda.device_count(),
        "logical_device": str(device),
        "logical_device_index": device.index,
        "supervisor_assigned_physical_gpu_uuid": (
            expected_physical_gpu_uuid
        ),
        "physical_gpu_index": physical_index,
        "physical_gpu_uuid": gpu_uuid,
        "physical_pci_bus_id": pci_bus_id,
        "driver_version": driver_version,
        "device_name": properties.name,
        "compute_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "total_memory_bytes": int(properties.total_memory),
    }


def _prepare_device_inputs(
    runs: Sequence[RunSpec],
    *,
    mask_replicate: int,
    device: torch.device,
) -> tuple[
    DeviceInputs,
    tuple[str, torch.Tensor],
    dict[str, object],
]:
    reference = runs[0]
    core = load_and_refit_full_core(
        _resolve_prepared_artifact(reference.config)
    )
    if core.checksums.preprocessing_sha256 != reference.preprocessing_sha256:
        raise SensitivityWorkflowError(
            "recreated preprocessing checksum mismatch"
        )
    tokens = tokenize_expression_counts(core.expression_counts)
    token_audit = audit_expression_tokens(tokens)
    if (
        token_audit.get("token_checksum_sha256")
        != reference.token_matrix_sha256
    ):
        raise SensitivityWorkflowError(
            "recreated token matrix checksum mismatch"
        )
    graph_config = _mapping(reference.config.get("graph"), "graph config")
    graph = _build_graph(core, graph_config)
    if (
        graph.checksums.graph_sha256 != reference.graph_sha256
        or graph.qc.n_directed_edges
        != int(graph_config.get("expected_directed_edges", -1))
        or not graph.qc.receiver_sorted
        or graph.qc.self_loops
        or graph.qc.duplicate_directed_edges
    ):
        raise SensitivityWorkflowError(
            "recreated graph identity/structure mismatch"
        )
    edge_index, edge_attributes = graph.concatenate()
    bundle = _recreate_masks(core, reference.mask_manifest)
    if bundle.checksum != reference.mask_bundle_sha256:
        raise SensitivityWorkflowError(
            "recreated mask bundle checksum mismatch"
        )
    selected = select_locked_whole_node_masks(bundle)
    if mask_replicate not in {0, 1, 2}:
        raise SensitivityWorkflowError("mask replicate must be 0, 1, or 2")
    entry_id, mask = selected[mask_replicate]
    expected_row = _whole_node_manifest_rows(
        reference.mask_manifest
    )[mask_replicate]
    if entry_id != expected_row["entry_id"]:
        raise SensitivityWorkflowError(
            "selected whole-node mask identity mismatch"
        )
    inputs = DeviceInputs(
        expression=torch.from_numpy(
            np.asarray(tokens, dtype=np.float32)
        ).to(device=device),
        node_covariates=torch.from_numpy(
            np.asarray(core.node_covariates, dtype=np.float32)
        ).to(device=device),
        edge_index=torch.from_numpy(
            np.asarray(edge_index, dtype=np.int64)
        ).to(device=device),
        edge_attributes=torch.from_numpy(
            np.asarray(edge_attributes, dtype=np.float32)
        ).to(device=device),
    )
    provenance = {
        "preprocessing_sha256": core.checksums.preprocessing_sha256,
        "token_matrix_sha256": token_audit["token_checksum_sha256"],
        "graph_sha256": graph.checksums.graph_sha256,
        "directed_edges": graph.qc.n_directed_edges,
        "mask_bundle_sha256": bundle.checksum,
        "mask_replicate": mask_replicate,
        "mask_entry_id": entry_id,
        "mask_checksum": expected_row["mask_checksum"],
        "n_target_nodes": int(whole_node_targets(mask).numel()),
        "shape": [core.n_nodes, core.n_genes],
    }
    return inputs, (entry_id, mask.to(device=device)), provenance


def _model_state_record(
    label: str,
    model: TokenizedReceiverChunkedEdgeConditionedGATv2,
    *,
    source: str,
    paired_seed: int | None,
) -> dict[str, object]:
    return {
        "label": label,
        "source": source,
        "paired_seed": paired_seed,
        "state_dict_sha256": _state_dict_sha256(model.state_dict()),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }


def _compute_shard_payload(
    *,
    shard_type: str,
    analysis_input_sha256: str,
    models: Mapping[str, TokenizedReceiverChunkedEdgeConditionedGATv2],
    model_records: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]],
    inputs: DeviceInputs,
    mask: tuple[str, torch.Tensor],
    data_provenance: Mapping[str, Any],
    device: torch.device,
    gpu_identity: Mapping[str, Any],
    control_seed: int | None,
    reviewed_resource_pilot_sha256: str,
    reviewed_identical_control_sha256: str | None,
) -> dict[str, object]:
    if set(models) != {label for pair in pairs for label in pair}:
        raise SensitivityWorkflowError(
            "shard models do not exactly match requested pairs"
        )
    device_models = {
        label: model.to(device=device).eval()
        for label, model in models.items()
    }
    weights = {
        label: observed_token_projection_weights(model.encoder).detach()
        for label, model in device_models.items()
    }
    entry_id, gene_mask = mask
    targets = whole_node_targets(gene_mask)
    observed = ~gene_mask
    statistics: dict[
        tuple[str, str], list[ProbeSufficientStatistics]
    ] = {pair: [] for pair in pairs}
    probe_records: list[Mapping[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    for probe_index in range(PROBES_PER_MASK):
        probe, probe_record = make_rademacher_probe(
            (int(targets.numel()), int(inputs.expression.shape[1]), 4),
            mask_entry_id=entry_id,
            probe_index=probe_index,
            device=device,
        )
        gradients = {
            label: preactivation_probe_vjp(
                model,
                input_expression=inputs.expression,
                gene_mask=gene_mask,
                edge_index=inputs.edge_index,
                edge_attributes=inputs.edge_attributes,
                node_covariates=inputs.node_covariates,
                probe=probe,
            )
            for label, model in device_models.items()
        }
        scored = multi_tangent_vjp_statistics(
            {
                label: (gradients[label], weights[label])
                for label in device_models
            },
            observed,
            pairs,
            mask_entry_id=entry_id,
            probe_index=probe_index,
            node_chunk_size=_NODE_PROJECTION_CHUNK_SIZE,
        )
        for pair in pairs:
            statistics[pair].append(scored[pair])
        probe_records.append(probe_record)
        print(
            json.dumps(
                {
                    "event": "categorical_sensitivity_probe_complete",
                    "shard_type": shard_type,
                    "mask_entry_id": entry_id,
                    "control_seed": control_seed,
                    "probe_index": probe_index,
                    "probe_count": PROBES_PER_MASK,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del gradients, scored, probe
    torch.cuda.synchronize(device)
    duration = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    for model in device_models.values():
        model.to(device="cpu")
    torch.cuda.empty_cache()
    return {
        "schema_version": _SHARD_SCHEMA_VERSION,
        "artifact_kind": "g2_token_categorical_sensitivity_shard",
        "protocol": PROTOCOL_VERSION,
        "analysis_mode": _ANALYSIS_MODE,
        "analysis_input_sha256": analysis_input_sha256,
        "shard_type": shard_type,
        "mask_replicate": data_provenance["mask_replicate"],
        "mask_entry_id": entry_id,
        "mask_checksum": data_provenance["mask_checksum"],
        "control_seed": control_seed,
        "reviewed_resource_pilot_sha256": (
            reviewed_resource_pilot_sha256
        ),
        "reviewed_identical_control_sha256": (
            reviewed_identical_control_sha256
        ),
        "models": [dict(record) for record in model_records],
        "pairs": [
            {"reference": pair[0], "candidate": pair[1]}
            for pair in pairs
        ],
        "probe_records": [dict(record) for record in probe_records],
        "statistics": {
            _pair_key(pair): [
                row.to_dict() for row in statistics[pair]
            ]
            for pair in pairs
        },
        "data_provenance": dict(data_provenance),
        "resource": {
            "device": str(device),
            "isolated_gpu": dict(gpu_identity),
            "floating_point": "fp32_no_amp_no_tf32",
            "float32_matmul_precision": (
                torch.get_float32_matmul_precision()
            ),
            "cuda_matmul_allow_tf32": (
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "duration_seconds": duration,
            "peak_cuda_memory_allocated_bytes": peak,
            "full_graph_vjp_count": len(models) * PROBES_PER_MASK,
        },
    }


def _pair_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__vs__{pair[1]}"


def _shard_path(
    work_root: Path,
    *,
    shard_type: str,
    mask_replicate: int,
    control_seed: int | None = None,
) -> Path:
    if mask_replicate not in {0, 1, 2}:
        raise SensitivityWorkflowError(
            "mask replicate must be 0, 1, or 2"
        )
    if shard_type in {"trained", "identical"}:
        return (
            work_root
            / "shards"
            / shard_type
            / f"mask_r{mask_replicate:03d}.json"
        )
    if shard_type == "random" and control_seed in _RANDOM_SEEDS:
        return (
            work_root
            / "shards"
            / "random"
            / f"seed_{control_seed}"
            / f"mask_r{mask_replicate:03d}.json"
        )
    raise SensitivityWorkflowError("invalid shard type/control seed")


def _pilot_path(work_root: Path) -> Path:
    return work_root / "resource_pilot.json"


def _identical_review_path(work_root: Path) -> Path:
    return work_root / "identical_control_review.json"


def _validate_reviewed_identical(
    work_root: Path,
    *,
    analysis_input_sha256: str,
    reviewed_sha256: str | None,
    reviewed_pilot_sha256: str,
) -> Mapping[str, Any]:
    path = _identical_review_path(work_root)
    payload = _verify_bound_json(
        path,
        analysis_input_sha256=analysis_input_sha256,
    )
    checks = _mapping(
        payload.get("checks"),
        "identical-control review checks",
    )
    if (
        reviewed_sha256 != _sha256_file(path)
        or payload.get("schema_version") != 1
        or payload.get("artifact_kind")
        != "g2_token_categorical_sensitivity_identical_control_review"
        or payload.get("protocol") != PROTOCOL_VERSION
        or payload.get("analysis_mode") != _ANALYSIS_MODE
        or payload.get("analysis_input_sha256")
        != analysis_input_sha256
        or payload.get("reviewed_resource_pilot_sha256")
        != reviewed_pilot_sha256
        or payload.get("identical_shard_count") != 3
        or payload.get("probe_count") != 3 * PROBES_PER_MASK
        or checks.get("cosine_ci_lower_at_least_0_9999") is not True
        or checks.get(
            "relative_discrepancy_ci_upper_at_most_0_001"
        ) is not True
        or checks.get("norm_ratio_ci_within_0_999_1_001") is not True
        or payload.get("passes") is not True
    ):
        raise SensitivityWorkflowError(
            "full shard requires an explicitly reviewed, passing "
            "identical-control review SHA"
        )
    return payload


def _validate_reviewed_pilot(
    work_root: Path,
    *,
    analysis_input_sha256: str,
    reviewed_sha256: str | None,
) -> Mapping[str, Any]:
    path = _pilot_path(work_root)
    payload = _verify_bound_json(
        path,
        analysis_input_sha256=analysis_input_sha256,
    )
    manifest = _load_json(work_root / "analysis_manifest.json")
    _validate_pilot_payload(payload, manifest=manifest)
    if (
        reviewed_sha256 != _sha256_file(path)
        or payload.get("passes") is not True
    ):
        raise SensitivityWorkflowError(
            "full shard requires explicitly reviewed matching "
            "--reviewed-pilot-sha256"
        )
    return payload


def _run_pilot(
    *,
    work_root: Path,
    analysis_input_sha256: str,
    runs: Sequence[RunSpec],
    device: torch.device,
    gpu_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    path = _pilot_path(work_root)
    with _ExclusiveLock(work_root / "locks" / "resource_pilot.lock"):
        if path.is_file():
            payload = _verify_bound_json(
                path,
                analysis_input_sha256=analysis_input_sha256,
            )
            _validate_pilot_payload(
                payload,
                manifest=_load_analysis_manifest(work_root),
            )
            return payload
        run = {item.label: item for item in runs}["wider_s0"]
        inputs, mask, provenance = _prepare_device_inputs(
            runs,
            mask_replicate=0,
            device=device,
        )
        model = run.instantiate_trained().to(device=device).eval()
        model_state = _model_state_record(
            "wider_s0",
            model,
            source=run.run_id,
            paired_seed=None,
        )
        weights = observed_token_projection_weights(model.encoder).detach()
        entry_id, gene_mask = mask
        targets = whole_node_targets(gene_mask)
        probe, probe_record = make_rademacher_probe(
            (int(targets.numel()), int(inputs.expression.shape[1]), 4),
            mask_entry_id=entry_id,
            probe_index=0,
            device=device,
        )
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        total_started = time.monotonic()
        vjp_started = time.monotonic()
        gradient = preactivation_probe_vjp(
            model,
            input_expression=inputs.expression,
            gene_mask=gene_mask,
            edge_index=inputs.edge_index,
            edge_attributes=inputs.edge_attributes,
            node_covariates=inputs.node_covariates,
            probe=probe,
        )
        torch.cuda.synchronize(device)
        vjp_seconds = time.monotonic() - vjp_started
        projection_started = time.monotonic()
        norm_stats = paired_tangent_vjp_statistics(
            gradient,
            weights,
            gradient,
            weights,
            ~gene_mask,
            mask_entry_id=entry_id,
            probe_index=0,
            node_chunk_size=_NODE_PROJECTION_CHUNK_SIZE,
        )
        torch.cuda.synchronize(device)
        projection_seconds = time.monotonic() - projection_started
        total_seconds = time.monotonic() - total_started
        peak = int(torch.cuda.max_memory_allocated(device))
        peak_gib = peak / float(1024**3)
        finite = (
            math.isfinite(norm_stats.reference_squared_norm)
            and norm_stats.reference_squared_norm > 0.0
            and math.isfinite(total_seconds)
            and total_seconds > 0.0
        )
        passes = finite and peak_gib <= 20.5
        payload: dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": "g2_token_categorical_sensitivity_resource_pilot",
            "protocol": PROTOCOL_VERSION,
            "analysis_mode": _ANALYSIS_MODE,
            "analysis_input_sha256": analysis_input_sha256,
            "model_label": "wider_s0",
            "model_run_id": run.run_id,
            "model_state": model_state,
            "mask_replicate": 0,
            "mask_entry_id": entry_id,
            "probe_index": 0,
            "probe": probe_record,
            "data_provenance": provenance,
            "vjp_wall_seconds": vjp_seconds,
            "tangent_projection_wall_seconds": projection_seconds,
            "total_model_equivalent_wall_seconds": total_seconds,
            "peak_cuda_memory_allocated_bytes": peak,
            "peak_cuda_memory_allocated_gib": peak_gib,
            "tangent_vjp_squared_norm": (
                norm_stats.reference_squared_norm
            ),
            "projected_2304_vjp_seconds": total_seconds * 2_304,
            "projected_2304_vjp_hours": (
                total_seconds * 2_304 / 3600.0
            ),
            "resource": {
                "device": str(device),
                "isolated_gpu": dict(gpu_identity),
                "floating_point": "fp32_no_amp_no_tf32",
                "float32_matmul_precision": (
                    torch.get_float32_matmul_precision()
                ),
                "cuda_matmul_allow_tf32": (
                    torch.backends.cuda.matmul.allow_tf32
                ),
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
            },
            "projection_is_not_a_time_gate": True,
            "criteria": {
                "finite_nonzero_vjp": finite,
                "peak_allocated_vram_at_most_20_5_gib": (
                    peak_gib <= 20.5
                ),
            },
            "passes": passes,
            "explicit_review_required_before_full_shards": True,
        }
        model.to(device="cpu")
        del model, gradient, weights, probe
        torch.cuda.empty_cache()
        _write_bound_json(
            path,
            payload,
            analysis_input_sha256=analysis_input_sha256,
        )
        if not passes:
            raise SensitivityWorkflowError(
                "mandatory resource pilot failed"
            )
        return payload


def _run_shard(
    *,
    work_root: Path,
    analysis_input_sha256: str,
    runs: Sequence[RunSpec],
    shard_type: str,
    mask_replicate: int,
    control_seed: int | None,
    reviewed_pilot_sha256: str | None,
    reviewed_identical_sha256: str | None,
    device: torch.device,
    gpu_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    _validate_reviewed_pilot(
        work_root,
        analysis_input_sha256=analysis_input_sha256,
        reviewed_sha256=reviewed_pilot_sha256,
    )
    assert reviewed_pilot_sha256 is not None
    if shard_type in {"trained", "random"}:
        _validate_reviewed_identical(
            work_root,
            analysis_input_sha256=analysis_input_sha256,
            reviewed_sha256=reviewed_identical_sha256,
            reviewed_pilot_sha256=reviewed_pilot_sha256,
        )
        assert reviewed_identical_sha256 is not None
    elif reviewed_identical_sha256 is not None:
        raise SensitivityWorkflowError(
            "identical shards cannot bind a pre-existing identical review"
        )
    path = _shard_path(
        work_root,
        shard_type=shard_type,
        mask_replicate=mask_replicate,
        control_seed=control_seed,
    )
    lock_name = path.relative_to(work_root).as_posix().replace("/", "__")
    with _ExclusiveLock(work_root / "locks" / f"{lock_name}.lock"):
        if path.is_file():
            payload = _verify_bound_json(
                path,
                analysis_input_sha256=analysis_input_sha256,
            )
            manifest = _load_analysis_manifest(work_root)
            common = _mapping(
                manifest.get("common_identity"),
                "common identity",
            )
            masks = common.get("whole_node_masks")
            if not isinstance(masks, Sequence) or len(masks) != 3:
                raise SensitivityWorkflowError(
                    "analysis manifest lacks whole-node masks"
                )
            mask_row = _mapping(
                masks[mask_replicate],
                "resumed shard mask row",
            )
            _validate_shard(
                payload,
                manifest=manifest,
                analysis_input_sha256=analysis_input_sha256,
                reviewed_resource_pilot_sha256=(
                    reviewed_pilot_sha256
                ),
                reviewed_identical_control_sha256=(
                    reviewed_identical_sha256
                ),
                shard_type=shard_type,
                mask_row=mask_row,
                control_seed=control_seed,
                expected_probe_records=_expected_probe_records(
                    mask_row,
                    common=common,
                ),
            )
            return payload
        by_label = {run.label: run for run in runs}
        inputs, mask, provenance = _prepare_device_inputs(
            runs,
            mask_replicate=mask_replicate,
            device=device,
        )
        if shard_type == "trained":
            models = {
                label: by_label[label].instantiate_trained()
                for label in sorted(by_label)
            }
            model_records = [
                _model_state_record(
                    label,
                    model,
                    source=by_label[label].run_id,
                    paired_seed=None,
                )
                for label, model in models.items()
            ]
            pairs = _TRAINED_PAIRS
        elif shard_type == "identical":
            source = by_label["current_s0"]
            models = {
                "identical_a": source.instantiate_trained(),
                "identical_b": source.instantiate_trained(),
            }
            model_records = [
                _model_state_record(
                    label,
                    model,
                    source=source.run_id,
                    paired_seed=None,
                )
                for label, model in models.items()
            ]
            pairs = (_IDENTICAL_PAIR,)
        elif shard_type == "random" and control_seed in _RANDOM_SEEDS:
            current = by_label["current_s0"]
            wider = by_label["wider_s0"]
            labels = (
                f"random_current_s{control_seed}",
                f"random_wider_s{control_seed}",
            )
            models = {
                labels[0]: current.instantiate_random(int(control_seed)),
                labels[1]: wider.instantiate_random(int(control_seed)),
            }
            model_records = [
                _model_state_record(
                    label,
                    model,
                    source="deterministic_reinitialization",
                    paired_seed=control_seed,
                )
                for label, model in models.items()
            ]
            pairs = (labels,)
        else:
            raise SensitivityWorkflowError("invalid shard request")
        payload = _compute_shard_payload(
            shard_type=shard_type,
            analysis_input_sha256=analysis_input_sha256,
            models=models,
            model_records=model_records,
            pairs=pairs,
            inputs=inputs,
            mask=mask,
            data_provenance=provenance,
            device=device,
            gpu_identity=gpu_identity,
            control_seed=control_seed,
            reviewed_resource_pilot_sha256=reviewed_pilot_sha256,
            reviewed_identical_control_sha256=(
                reviewed_identical_sha256
            ),
        )
        _write_bound_json(
            path,
            payload,
            analysis_input_sha256=analysis_input_sha256,
        )
        return payload


def _statistics_from_mapping(
    value: Mapping[str, Any],
) -> ProbeSufficientStatistics:
    try:
        return ProbeSufficientStatistics(
            mask_entry_id=str(value.get("mask_entry_id", "")),
            probe_index=_integer(value.get("probe_index"), "probe index"),
            reference_squared_norm=_finite(
                value.get("reference_squared_norm"),
                "reference squared norm",
            ),
            candidate_squared_norm=_finite(
                value.get("candidate_squared_norm"),
                "candidate squared norm",
            ),
            cross_inner_product=_finite(
                value.get("cross_inner_product"),
                "cross inner product",
            ),
        )
    except ValueError as exc:
        raise SensitivityWorkflowError(
            "invalid probe sufficient statistics"
        ) from exc


def _expected_pairs(
    shard_type: str,
    control_seed: int | None,
) -> tuple[tuple[str, str], ...]:
    if shard_type == "trained":
        return _TRAINED_PAIRS
    if shard_type == "identical":
        return (_IDENTICAL_PAIR,)
    if shard_type == "random" and control_seed in _RANDOM_SEEDS:
        return (
            (
                f"random_current_s{control_seed}",
                f"random_wider_s{control_seed}",
            ),
        )
    raise SensitivityWorkflowError("invalid expected shard pair request")


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _is_gpu_uuid(value: object) -> bool:
    text = str(value)
    if not text.startswith("GPU-"):
        return False
    try:
        return str(uuid.UUID(text[4:])) == text[4:].lower()
    except ValueError:
        return False


def _analysis_run_rows(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw_rows = manifest.get("runs")
    if not isinstance(raw_rows, Sequence) or len(raw_rows) != 6:
        raise SensitivityWorkflowError(
            "analysis manifest must bind exactly six runs"
        )
    rows: dict[str, Mapping[str, Any]] = {}
    for raw_row in raw_rows:
        row = _mapping(raw_row, "analysis run row")
        width = str(row.get("width_label", ""))
        seed = _integer(row.get("seed"), "analysis run seed")
        label = f"{width}_s{seed}"
        expected_variant = (
            _CURRENT_VARIANT if width == "current" else _WIDER_VARIANT
        )
        if (
            label in rows
            or width not in {"current", "wider"}
            or seed not in {0, 1, 2}
            or row.get("variant_label") != expected_variant
            or not str(row.get("run_id", "")).startswith("r_")
            or not Path(str(row.get("run_path", ""))).is_absolute()
            or not _is_sha256(row.get("state_dict_sha256"))
            or not _is_sha256(row.get("checkpoint_sha256"))
            or any(
                not _is_sha256(row.get(field))
                for field in (
                    "run_bundle_checksum_manifest_sha256",
                    "completion_marker_sha256",
                    "config_resolved_sha256",
                    "final_metrics_sha256",
                    "fixed_mask_provenance_sha256",
                    "tokenization_provenance_sha256",
                )
            )
            or _integer(
                row.get("parameter_count"),
                f"{label} parameter count",
            )
            <= 0
        ):
            raise SensitivityWorkflowError(
                "analysis run provenance is invalid"
            )
        verification = _mapping(
            row.get("standard_bundle_verification"),
            f"{label} bundle verification",
        )
        if (
            verification.get("valid") is not True
            or verification.get("status") != "success"
            or _integer(
                verification.get("file_count"),
                f"{label} bundle file count",
            )
            <= 0
        ):
            raise SensitivityWorkflowError(
                "analysis run was not checksum-verified"
            )
        rows[label] = row
    expected = {
        *(f"current_s{seed}" for seed in range(3)),
        *(f"wider_s{seed}" for seed in range(3)),
    }
    if set(rows) != expected:
        raise SensitivityWorkflowError(
            "analysis manifest run inventory is incomplete"
        )
    return rows


def _expected_model_records(
    manifest: Mapping[str, Any],
    *,
    shard_type: str,
    control_seed: int | None,
) -> tuple[dict[str, object], ...]:
    runs = _analysis_run_rows(manifest)
    if shard_type == "trained":
        return tuple(
            {
                "label": label,
                "source": runs[label]["run_id"],
                "paired_seed": None,
                "state_dict_sha256": runs[label]["state_dict_sha256"],
                "parameter_count": runs[label]["parameter_count"],
            }
            for label in sorted(runs)
        )
    if shard_type == "identical":
        source = runs["current_s0"]
        return tuple(
            {
                "label": label,
                "source": source["run_id"],
                "paired_seed": None,
                "state_dict_sha256": source["state_dict_sha256"],
                "parameter_count": source["parameter_count"],
            }
            for label in _IDENTICAL_PAIR
        )
    if shard_type == "random" and control_seed in _RANDOM_SEEDS:
        return (
            {
                "label": f"random_current_s{control_seed}",
                "source": "deterministic_reinitialization",
                "paired_seed": control_seed,
                "state_dict_sha256": None,
                "parameter_count": runs["current_s0"]["parameter_count"],
            },
            {
                "label": f"random_wider_s{control_seed}",
                "source": "deterministic_reinitialization",
                "paired_seed": control_seed,
                "state_dict_sha256": None,
                "parameter_count": runs["wider_s0"]["parameter_count"],
            },
        )
    raise SensitivityWorkflowError("invalid model-record request")


def _validate_model_records(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    shard_type: str,
    control_seed: int | None,
) -> bytes:
    raw_records = payload.get("models")
    expected = _expected_model_records(
        manifest,
        shard_type=shard_type,
        control_seed=control_seed,
    )
    if (
        not isinstance(raw_records, Sequence)
        or len(raw_records) != len(expected)
    ):
        raise SensitivityWorkflowError(
            "shard model inventory is incomplete"
        )
    normalized: list[dict[str, object]] = []
    for raw_record, expected_record in zip(raw_records, expected, strict=True):
        record = _mapping(raw_record, "shard model record")
        state_sha = record.get("state_dict_sha256")
        if (
            record.get("label") != expected_record["label"]
            or record.get("source") != expected_record["source"]
            or record.get("paired_seed") != expected_record["paired_seed"]
            or _integer(
                record.get("parameter_count"),
                "model parameter count",
            )
            != expected_record["parameter_count"]
            or not _is_sha256(state_sha)
            or (
                expected_record["state_dict_sha256"] is not None
                and state_sha != expected_record["state_dict_sha256"]
            )
        ):
            raise SensitivityWorkflowError(
                "shard model state provenance mismatch"
            )
        normalized.append(
            {
                "label": record["label"],
                "source": record["source"],
                "paired_seed": record["paired_seed"],
                "state_dict_sha256": state_sha,
                "parameter_count": record["parameter_count"],
            }
        )
    if (
        shard_type == "identical"
        and normalized[0]["state_dict_sha256"]
        != normalized[1]["state_dict_sha256"]
    ):
        raise SensitivityWorkflowError(
            "independent identical loads do not contain identical states"
        )
    if (
        shard_type == "random"
        and normalized[0]["state_dict_sha256"]
        == normalized[1]["state_dict_sha256"]
    ):
        raise SensitivityWorkflowError(
            "different-width random controls have identical state digests"
        )
    return _canonical_json(normalized)


def _expected_probe_shape(
    mask_row: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
) -> list[int]:
    shape = list(mask_row.get("shape", ()))
    expected_shape = [
        _integer(common.get("n_nodes"), "common node count"),
        _integer(common.get("n_genes"), "common gene count"),
    ]
    if shape != expected_shape:
        raise SensitivityWorkflowError(
            "mask shape disagrees with common data identity"
        )
    n_masked = _integer(mask_row.get("n_masked"), "mask entry count")
    if (
        n_masked <= 0
        or n_masked % expected_shape[1] != 0
        or n_masked // expected_shape[1] > expected_shape[0]
    ):
        raise SensitivityWorkflowError(
            "whole-node mask summary is inconsistent"
        )
    return [n_masked // expected_shape[1], expected_shape[1], 4]


def _validate_data_provenance(
    value: object,
    *,
    common: Mapping[str, Any],
    mask_row: Mapping[str, Any],
) -> None:
    provenance = _mapping(value, "shard data provenance")
    expected_probe_shape = _expected_probe_shape(mask_row, common=common)
    if (
        provenance.get("preprocessing_sha256")
        != common.get("preprocessing_sha256")
        or provenance.get("token_matrix_sha256")
        != common.get("token_matrix_sha256")
        or provenance.get("graph_sha256") != common.get("graph_sha256")
        or provenance.get("mask_bundle_sha256")
        != common.get("mask_bundle_sha256")
        or provenance.get("directed_edges") != common.get("directed_edges")
        or provenance.get("mask_replicate") != mask_row.get("replicate")
        or provenance.get("mask_entry_id") != mask_row.get("entry_id")
        or provenance.get("mask_checksum") != mask_row.get("mask_checksum")
        or provenance.get("shape")
        != [common.get("n_nodes"), common.get("n_genes")]
        or provenance.get("n_target_nodes") != expected_probe_shape[0]
    ):
        raise SensitivityWorkflowError(
            "shard graph/token/mask provenance mismatch"
        )


def _validate_isolated_gpu_execution(value: object) -> Mapping[str, Any]:
    resource = _mapping(value, "GPU execution resource")
    gpu = _mapping(resource.get("isolated_gpu"), "isolated GPU")
    if (
        resource.get("device") != "cuda:0"
        or gpu.get("visible_device_count") != 1
        or gpu.get("logical_device") != "cuda:0"
        or gpu.get("logical_device_index") != 0
        or not str(gpu.get("cuda_visible_devices", ""))
        or "," in str(gpu.get("cuda_visible_devices", ""))
        or gpu.get("physical_gpu_index") not in {5, 6, 7}
        or gpu.get("supervisor_assigned_physical_gpu_uuid")
        != gpu.get("physical_gpu_uuid")
        or not _is_gpu_uuid(gpu.get("physical_gpu_uuid"))
        or not str(gpu.get("physical_pci_bus_id", ""))
        or not str(gpu.get("driver_version", ""))
        or not str(gpu.get("device_name", ""))
        or (
            not isinstance(gpu.get("compute_capability"), Sequence)
            or len(gpu["compute_capability"]) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in gpu["compute_capability"]
            )
        )
        or _integer(
            gpu.get("total_memory_bytes"),
            "physical GPU memory",
        )
        <= 0
        or resource.get("floating_point") != "fp32_no_amp_no_tf32"
        or resource.get("float32_matmul_precision") != "highest"
        or resource.get("cuda_matmul_allow_tf32") is not False
        or resource.get("cudnn_allow_tf32") is not False
        or resource.get("cublas_workspace_config") != ":4096:8"
    ):
        raise SensitivityWorkflowError(
            "isolated GPU/FP32 execution record is invalid"
        )
    return resource


def _validate_resource_record(
    value: object,
    *,
    shard_type: str,
    expected_physical_gpu_index: int,
) -> None:
    resource = _validate_isolated_gpu_execution(value)
    gpu = _mapping(resource.get("isolated_gpu"), "isolated GPU")
    expected_models = 6 if shard_type == "trained" else 2
    if (
        gpu.get("physical_gpu_index") != expected_physical_gpu_index
        or _finite(resource.get("duration_seconds"), "shard duration")
        <= 0.0
        or _integer(
            resource.get("peak_cuda_memory_allocated_bytes"),
            "peak CUDA bytes",
        )
        < 0
        or _integer(
            resource.get("full_graph_vjp_count"),
            "full-graph VJP count",
        )
        != expected_models * PROBES_PER_MASK
    ):
        raise SensitivityWorkflowError(
            "shard resource/execution record is invalid"
        )


def _validate_probe_records(
    value: object,
    *,
    mask_row: Mapping[str, Any],
    common: Mapping[str, Any],
    expected_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[int, Mapping[str, Any]]:
    records = value
    if not isinstance(records, Sequence) or len(records) != PROBES_PER_MASK:
        raise SensitivityWorkflowError("shard probe manifest is incomplete")
    expected_shape = _expected_probe_shape(mask_row, common=common)
    probe_by_index: dict[int, Mapping[str, Any]] = {}
    for raw_record in records:
        record = _mapping(raw_record, "probe record")
        index = _integer(record.get("probe_index"), "probe index")
        if (
            index in probe_by_index
            or record.get("mask_entry_id") != mask_row["entry_id"]
            or record.get("seed")
            != locked_seed(
                PROTOCOL_BASE_SEED,
                mask_row["entry_id"],
                index,
            )
            or record.get("shape") != expected_shape
            or record.get("generation_dtype") != "int8"
            or not _is_sha256(record.get("checksum_sha256"))
        ):
            raise SensitivityWorkflowError("shard probe record is invalid")
        probe_by_index[index] = record
    if set(probe_by_index) != set(range(PROBES_PER_MASK)):
        raise SensitivityWorkflowError("shard probe indices are incomplete")
    if expected_records is not None and _canonical_json(
        [probe_by_index[index] for index in range(PROBES_PER_MASK)]
    ) != _canonical_json(list(expected_records)):
        raise SensitivityWorkflowError(
            "probe checksums do not match deterministic CPU PCG64 probes"
        )
    return probe_by_index


def _expected_probe_records(
    mask_row: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    shape = _expected_probe_shape(mask_row, common=common)
    records: list[Mapping[str, Any]] = []
    for probe_index in range(PROBES_PER_MASK):
        probe, record = make_rademacher_probe(
            shape,
            mask_entry_id=str(mask_row["entry_id"]),
            probe_index=probe_index,
            device="cpu",
        )
        records.append(record)
        del probe
    return tuple(records)


def _validate_pilot_payload(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> None:
    analysis_sha = str(manifest.get("analysis_input_sha256", ""))
    common = _mapping(manifest.get("common_identity"), "common identity")
    masks = common.get("whole_node_masks")
    if not isinstance(masks, Sequence) or len(masks) != 3:
        raise SensitivityWorkflowError(
            "analysis manifest lacks pilot mask identity"
        )
    mask_row = _mapping(masks[0], "pilot mask row")
    runs = _analysis_run_rows(manifest)
    wider = runs["wider_s0"]
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind")
        != "g2_token_categorical_sensitivity_resource_pilot"
        or payload.get("protocol") != PROTOCOL_VERSION
        or payload.get("analysis_mode") != _ANALYSIS_MODE
        or payload.get("analysis_input_sha256") != analysis_sha
        or payload.get("model_label") != "wider_s0"
        or payload.get("model_run_id") != wider["run_id"]
        or payload.get("mask_replicate") != 0
        or payload.get("mask_entry_id") != mask_row["entry_id"]
        or payload.get("probe_index") != 0
    ):
        raise SensitivityWorkflowError(
            "resource pilot identity/provenance mismatch"
        )
    model_state = _mapping(payload.get("model_state"), "pilot model state")
    if model_state != {
        "label": "wider_s0",
        "source": wider["run_id"],
        "paired_seed": None,
        "state_dict_sha256": wider["state_dict_sha256"],
        "parameter_count": wider["parameter_count"],
    }:
        raise SensitivityWorkflowError(
            "resource pilot checkpoint state mismatch"
        )
    _validate_data_provenance(
        payload.get("data_provenance"),
        common=common,
        mask_row=mask_row,
    )
    expected_shape = _expected_probe_shape(mask_row, common=common)
    probe, expected_probe = make_rademacher_probe(
        expected_shape,
        mask_entry_id=str(mask_row["entry_id"]),
        probe_index=0,
        device="cpu",
    )
    del probe
    if _canonical_json(payload.get("probe")) != _canonical_json(
        expected_probe
    ):
        raise SensitivityWorkflowError(
            "resource pilot probe checksum mismatch"
        )
    pilot_resource = _validate_isolated_gpu_execution(
        payload.get("resource")
    )
    pilot_gpu = _mapping(
        pilot_resource.get("isolated_gpu"),
        "pilot isolated GPU",
    )
    vjp_seconds = _finite(
        payload.get("vjp_wall_seconds"),
        "pilot VJP duration",
    )
    projection_seconds = _finite(
        payload.get("tangent_projection_wall_seconds"),
        "pilot projection duration",
    )
    total_seconds = _finite(
        payload.get("total_model_equivalent_wall_seconds"),
        "pilot total duration",
    )
    peak = _integer(
        payload.get("peak_cuda_memory_allocated_bytes"),
        "pilot peak CUDA bytes",
    )
    peak_gib = _finite(
        payload.get("peak_cuda_memory_allocated_gib"),
        "pilot peak CUDA GiB",
    )
    norm = _finite(
        payload.get("tangent_vjp_squared_norm"),
        "pilot tangent VJP norm",
    )
    criteria = _mapping(payload.get("criteria"), "pilot criteria")
    if (
        vjp_seconds <= 0.0
        or pilot_gpu.get("physical_gpu_index") != 7
        or projection_seconds <= 0.0
        or total_seconds <= 0.0
        or peak < 0
        or norm <= 0.0
        or not math.isclose(
            peak_gib,
            peak / float(1024**3),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(
                payload.get("projected_2304_vjp_seconds"),
                "pilot projected seconds",
            ),
            total_seconds * 2_304,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _finite(
                payload.get("projected_2304_vjp_hours"),
                "pilot projected hours",
            ),
            total_seconds * 2_304 / 3600.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or payload.get("projection_is_not_a_time_gate") is not True
        or criteria.get("finite_nonzero_vjp") is not True
        or criteria.get(
            "peak_allocated_vram_at_most_20_5_gib"
        ) is not True
        or peak_gib > 20.5
        or payload.get("passes") is not True
        or payload.get(
            "explicit_review_required_before_full_shards"
        ) is not True
    ):
        raise SensitivityWorkflowError(
            "resource pilot criteria or measurements are invalid"
        )


def _validate_shared_trained_norms(
    rows_by_pair: Mapping[
        tuple[str, str],
        Sequence[ProbeSufficientStatistics],
    ],
) -> None:
    norms: dict[tuple[str, int], float] = {}
    for pair, rows in rows_by_pair.items():
        for row in rows:
            for label, value in (
                (pair[0], row.reference_squared_norm),
                (pair[1], row.candidate_squared_norm),
            ):
                key = (label, row.probe_index)
                if key in norms and norms[key] != value:
                    raise SensitivityWorkflowError(
                        "trained shard does not reuse each model VJP "
                        "consistently across pairs"
                    )
                norms[key] = value


def _validate_shard(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    analysis_input_sha256: str,
    reviewed_resource_pilot_sha256: str,
    reviewed_identical_control_sha256: str | None,
    shard_type: str,
    mask_row: Mapping[str, Any],
    control_seed: int | None,
    expected_probe_records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[
    dict[tuple[str, str], list[ProbeSufficientStatistics]],
    bytes,
]:
    if (
        payload.get("schema_version") != _SHARD_SCHEMA_VERSION
        or payload.get("artifact_kind")
        != "g2_token_categorical_sensitivity_shard"
        or payload.get("protocol") != PROTOCOL_VERSION
        or payload.get("analysis_mode") != _ANALYSIS_MODE
        or payload.get("analysis_input_sha256")
        != analysis_input_sha256
        or payload.get("shard_type") != shard_type
        or payload.get("mask_replicate") != mask_row["replicate"]
        or payload.get("mask_entry_id") != mask_row["entry_id"]
        or payload.get("mask_checksum") != mask_row["mask_checksum"]
        or payload.get("control_seed") != control_seed
        or payload.get("reviewed_resource_pilot_sha256")
        != reviewed_resource_pilot_sha256
        or payload.get("reviewed_identical_control_sha256")
        != reviewed_identical_control_sha256
    ):
        raise SensitivityWorkflowError(
            "shard identity/provenance contract mismatch"
        )
    pairs = _expected_pairs(shard_type, control_seed)
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, Sequence):
        raise SensitivityWorkflowError("shard pair list is missing")
    observed_pairs = tuple(
        (
            str(_mapping(row, "shard pair").get("reference", "")),
            str(_mapping(row, "shard pair").get("candidate", "")),
        )
        for row in raw_pairs
    )
    if observed_pairs != pairs:
        raise SensitivityWorkflowError("shard pair definitions mismatch")
    common = _mapping(manifest.get("common_identity"), "common identity")
    _validate_data_provenance(
        payload.get("data_provenance"),
        common=common,
        mask_row=mask_row,
    )
    _validate_resource_record(
        payload.get("resource"),
        shard_type=shard_type,
        expected_physical_gpu_index=5 + int(mask_row["replicate"]),
    )
    _validate_probe_records(
        payload.get("probe_records"),
        mask_row=mask_row,
        common=common,
        expected_records=expected_probe_records,
    )
    model_identity = _validate_model_records(
        payload,
        manifest=manifest,
        shard_type=shard_type,
        control_seed=control_seed,
    )
    raw_stats = _mapping(payload.get("statistics"), "shard statistics")
    if set(raw_stats) != {_pair_key(pair) for pair in pairs}:
        raise SensitivityWorkflowError(
            "shard statistics contain missing or unexpected pairs"
        )
    result: dict[tuple[str, str], list[ProbeSufficientStatistics]] = {}
    expected_grid = {
        (str(mask_row["entry_id"]), index)
        for index in range(PROBES_PER_MASK)
    }
    for pair in pairs:
        raw_rows = raw_stats.get(_pair_key(pair))
        if not isinstance(raw_rows, Sequence):
            raise SensitivityWorkflowError(
                f"shard lacks statistics for {_pair_key(pair)}"
            )
        rows = [
            _statistics_from_mapping(_mapping(row, "statistic row"))
            for row in raw_rows
        ]
        if {
            (row.mask_entry_id, row.probe_index) for row in rows
        } != expected_grid or len(rows) != PROBES_PER_MASK:
            raise SensitivityWorkflowError(
                "shard statistic grid is incomplete"
            )
        result[pair] = rows
    if shard_type == "trained":
        _validate_shared_trained_norms(result)
    return result, model_identity


def _estimate_rows(
    rows: Sequence[ProbeSufficientStatistics],
    *,
    pair: tuple[str, str],
    mask_ids: Sequence[str],
) -> PairEstimate:
    return estimate_pair(
        rows,
        reference_label=pair[0],
        candidate_label=pair[1],
        mask_entry_ids=mask_ids,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )


def review_identical_shards(
    work_root: str | Path,
    *,
    reviewed_pilot_sha256: str,
) -> Mapping[str, Any]:
    """CPU-only fail-fast review of the exact three identical shards."""

    work = Path(work_root).resolve()
    with _ExclusiveLock(work / "locks" / "review-identical.lock"):
        manifest = _load_analysis_manifest(work)
        analysis_sha = str(manifest["analysis_input_sha256"])
        _validate_reviewed_pilot(
            work,
            analysis_input_sha256=analysis_sha,
            reviewed_sha256=reviewed_pilot_sha256,
        )
        path = _identical_review_path(work)
        existing: Mapping[str, Any] | None = None
        if path.is_file():
            existing = _verify_bound_json(
                path,
                analysis_input_sha256=analysis_sha,
            )
            if (
                existing.get("reviewed_resource_pilot_sha256")
                != reviewed_pilot_sha256
            ):
                raise SensitivityWorkflowError(
                    "existing identical review binds another pilot"
                )

        common = _mapping(
            manifest.get("common_identity"),
            "common identity",
        )
        raw_masks = common.get("whole_node_masks")
        if not isinstance(raw_masks, Sequence) or len(raw_masks) != 3:
            raise SensitivityWorkflowError(
                "analysis manifest lacks three whole-node masks"
            )
        masks = [
            _mapping(row, "whole-node mask row") for row in raw_masks
        ]
        expected_paths = [
            _shard_path(
                work,
                shard_type="identical",
                mask_replicate=replicate,
            )
            for replicate in range(3)
        ]
        _require_exact_bound_inventory(
            work / "shards" / "identical",
            expected_paths,
        )
        all_rows: list[ProbeSufficientStatistics] = []
        inventory: list[dict[str, object]] = []
        common_model_identity: bytes | None = None
        for replicate, (mask_row, shard_path) in enumerate(
            zip(masks, expected_paths, strict=True)
        ):
            expected_probes = _expected_probe_records(
                mask_row,
                common=common,
            )
            shard = _verify_bound_json(
                shard_path,
                analysis_input_sha256=analysis_sha,
            )
            rows, model_identity = _validate_shard(
                shard,
                manifest=manifest,
                analysis_input_sha256=analysis_sha,
                reviewed_resource_pilot_sha256=reviewed_pilot_sha256,
                reviewed_identical_control_sha256=None,
                shard_type="identical",
                mask_row=mask_row,
                control_seed=None,
                expected_probe_records=expected_probes,
            )
            if common_model_identity is None:
                common_model_identity = model_identity
            elif common_model_identity != model_identity:
                raise SensitivityWorkflowError(
                    "identical model state changed across mask shards"
                )
            all_rows.extend(rows[_IDENTICAL_PAIR])
            inventory.append(
                {
                    "path": str(shard_path),
                    "sha256": _sha256_file(shard_path),
                    "mask_replicate": replicate,
                    "mask_entry_id": mask_row["entry_id"],
                    "full_graph_vjp_count": 2 * PROBES_PER_MASK,
                }
            )
        estimate = _estimate_rows(
            all_rows,
            pair=_IDENTICAL_PAIR,
            mask_ids=[str(row["entry_id"]) for row in masks],
        )
        checks = {
            "cosine_ci_lower_at_least_0_9999": (
                estimate.cosine_interval.lower >= 0.9999
            ),
            "relative_discrepancy_ci_upper_at_most_0_001": (
                estimate.relative_discrepancy_interval.upper <= 0.001
            ),
            "norm_ratio_ci_within_0_999_1_001": (
                estimate.norm_ratio_interval.lower >= 0.999
                and estimate.norm_ratio_interval.upper <= 1.001
            ),
        }
        payload: dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": (
                "g2_token_categorical_sensitivity_identical_control_review"
            ),
            "protocol": PROTOCOL_VERSION,
            "analysis_mode": _ANALYSIS_MODE,
            "analysis_input_sha256": analysis_sha,
            "reviewed_resource_pilot_sha256": reviewed_pilot_sha256,
            "identical_shard_count": 3,
            "probe_count": 3 * PROBES_PER_MASK,
            "full_graph_vjp_count": 3 * 2 * PROBES_PER_MASK,
            "shard_inventory": inventory,
            "estimate": estimate.to_dict(),
            "interval_interpretation": (
                "95% technical mask-plus-Rademacher-trace-probe "
                "bootstrap variability; not biological or training-seed "
                "uncertainty"
            ),
            "checks": checks,
            "passes": all(checks.values()),
            "explicit_review_required_before_trained_or_random_shards": True,
        }
        if existing is None:
            _write_bound_json(
                path,
                payload,
                analysis_input_sha256=analysis_sha,
            )
        elif _canonical_json(existing) != _canonical_json(payload):
            raise SensitivityWorkflowError(
                "existing identical review does not match a fresh semantic "
                "revalidation of its shards"
            )
        if not payload["passes"]:
            raise SensitivityWorkflowError(
                "identical-control numerical review failed; trained and "
                "random shards are blocked"
            )
        return payload


def _pair_estimate_rows(
    estimates: Mapping[str, Mapping[str, PairEstimate] | PairEstimate],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, raw in estimates.items():
        values = {"identical": raw} if isinstance(raw, PairEstimate) else raw
        for label, estimate in values.items():
            assert isinstance(estimate, PairEstimate)
            rows.append(
                {
                    "family": family,
                    "label": label,
                    "interval_scope": (
                        "technical_mask_plus_rademacher_trace_probe"
                    ),
                    "reference": estimate.reference_label,
                    "candidate": estimate.candidate_label,
                    "cosine": estimate.point.cosine,
                    "cosine_ci_lower": estimate.cosine_interval.lower,
                    "cosine_ci_upper": estimate.cosine_interval.upper,
                    "relative_discrepancy": (
                        estimate.point.relative_discrepancy
                    ),
                    "relative_discrepancy_ci_lower": (
                        estimate.relative_discrepancy_interval.lower
                    ),
                    "relative_discrepancy_ci_upper": (
                        estimate.relative_discrepancy_interval.upper
                    ),
                    "norm_ratio": estimate.point.norm_ratio,
                    "norm_ratio_ci_lower": (
                        estimate.norm_ratio_interval.lower
                    ),
                    "norm_ratio_ci_upper": (
                        estimate.norm_ratio_interval.upper
                    ),
                }
            )
    return rows


def _csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _report_markdown(result: Mapping[str, Any]) -> str:
    decision = _mapping(result["decision"], "decision")
    primary = _mapping(result["estimates"]["primary"], "primary estimates")
    lines = [
        "# Post-hoc tokenized G2 relaxed categorical sensitivity",
        "",
        (
            "The prespecified 95% accuracy gate failed for all wider seeds. "
            "This analysis was run only after an explicit user request and is "
            "post-hoc exploratory; it does not revise the negative width or "
            "failed-gate conclusions."
        ),
        "",
        "## Numerical outcome",
        "",
        (
            "All 95% intervals below quantify only technical variation from "
            "resampling the three masks and Rademacher trace probes. They are "
            "not biological, patient, or training-seed uncertainty intervals."
        ),
        "",
        (
            f"- Identical-checkpoint numerical control: "
            f"**{'PASS' if decision['analysis_numerically_valid'] else 'FAIL'}**."
        ),
        (
            f"- Exploratory operational cross-width match: "
            f"**{'PASS' if decision['operational_match'] else 'FAIL'}**."
        ),
        "",
        "| Seed | Cosine [95% technical CI] | Relative discrepancy "
        "[95% technical CI] | Norm ratio wider/current "
        "[95% technical CI] | All criteria |",
        "|---:|---:|---:|---:|:---:|",
    ]
    checks = _mapping(decision["primary"], "primary checks")
    for seed in range(3):
        estimate = _mapping(primary[str(seed)], f"primary seed {seed}")
        point = _mapping(estimate["point"], "point")
        intervals = _mapping(estimate["intervals"], "intervals")

        def formatted(name: str) -> str:
            interval = _mapping(intervals[name], f"{name} interval")
            return (
                f"{float(point[name]):.4f} "
                f"[{float(interval['lower']):.4f}, "
                f"{float(interval['upper']):.4f}]"
            )

        passed = _mapping(checks[str(seed)], "seed checks")["passed"]
        lines.append(
            f"| {seed} | {formatted('cosine')} | "
            f"{formatted('relative_discrepancy')} | "
            f"{formatted('norm_ratio')} | "
            f"{'yes' if passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "- Gradients are local, model- and scale-dependent "
                "sensitivities on one transductive core. They are not "
                "biological interactions or causal effects."
            ),
            (
                "- Random current/wider controls use the same paired seed. "
                "Aligned parameter shapes may share RNG prefixes, raising "
                "null similarity and making separation conservative."
            ),
            (
                "- Poor class-balanced and nonzero token accuracy remains the "
                "strongest limitation; functional similarity cannot rescue "
                "an uninformative predictor."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_output(
    output: Path,
    *,
    result: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, object]],
    shard_paths: Sequence[Path],
    analysis_manifest_path: Path,
    pilot_path: Path,
    identical_review_path: Path,
) -> None:
    if output.exists():
        raise SensitivityWorkflowError(
            f"refusing to overwrite output {output}"
        )
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise SensitivityWorkflowError(
            f"staging path already exists: {staging}"
        )
    staging.mkdir(parents=True)
    try:
        (staging / "comparison.json").write_bytes(
            _canonical_json(result) + b"\n"
        )
        (staging / "pair_estimates.csv").write_text(
            _csv_text(pair_rows),
            encoding="utf-8",
        )
        (staging / "report.md").write_text(
            _report_markdown(result),
            encoding="utf-8",
        )
        (staging / "protocol.json").write_bytes(
            _canonical_json(locked_protocol_record()) + b"\n"
        )
        shutil.copy2(
            analysis_manifest_path,
            staging / "analysis_manifest.json",
        )
        shutil.copy2(pilot_path, staging / pilot_path.name)
        shutil.copy2(
            _sidecar_path(pilot_path),
            staging / _sidecar_path(pilot_path).name,
        )
        shutil.copy2(
            identical_review_path,
            staging / identical_review_path.name,
        )
        shutil.copy2(
            _sidecar_path(identical_review_path),
            staging / _sidecar_path(identical_review_path).name,
        )
        raw = staging / "shards"
        raw.mkdir()
        for index, path in enumerate(shard_paths):
            target = raw / f"{index:02d}_{path.parent.name}_{path.name}"
            shutil.copy2(path, target)
            shutil.copy2(
                _sidecar_path(path),
                raw / f"{target.name}.sha256.json",
            )
        files = {
            path.relative_to(staging).as_posix(): _sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        owner = {
            "schema_version": 1,
            "artifact_kind": (
                "g2_token_categorical_sensitivity_post_hoc_comparison"
            ),
            "files": files,
            "content_sha256": hashlib.sha256(
                _canonical_json(files)
            ).hexdigest(),
        }
        (staging / ".g2-token-sensitivity-owner.json").write_bytes(
            _canonical_json(owner) + b"\n"
        )
        # Validate every staged content checksum before creating the
        # completion marker. The subsequently renamed directory is the
        # atomic publication unit.
        _verify_output_content(staging)
        completion = {
            "schema_version": 1,
            "status": "success",
            "artifact_kind": (
                "g2_token_categorical_sensitivity_post_hoc_comparison"
            ),
            "analysis_input_sha256": result["analysis_input_sha256"],
            "content_sha256": owner["content_sha256"],
            "owner_sha256": _sha256_file(
                staging / ".g2-token-sensitivity-owner.json"
            ),
            "verified_shard_count": result["verified_shard_count"],
            "verified_full_graph_vjp_count": (
                result["verified_full_graph_vjp_count"]
            ),
        }
        (staging / "_SUCCESS").write_bytes(
            _canonical_json(completion) + b"\n"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        _verify_output(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_output(output: Path) -> None:
    owner_path = output / ".g2-token-sensitivity-owner.json"
    completion_path = output / "_SUCCESS"
    owner, content_sha = _verify_output_content(output)
    completion = _load_json(completion_path)

    if (
        completion.get("status") != "success"
        or completion.get("artifact_kind")
        != "g2_token_categorical_sensitivity_post_hoc_comparison"
        or completion.get("content_sha256") != content_sha
        or completion.get("owner_sha256") != _sha256_file(owner_path)
        or completion.get("verified_shard_count") != 30
        or completion.get("verified_full_graph_vjp_count") != 2_304
    ):
        raise SensitivityWorkflowError(
            "output completion marker verification failed"
        )


def _verify_output_content(
    output: Path,
) -> tuple[Mapping[str, Any], str]:
    owner_path = output / ".g2-token-sensitivity-owner.json"
    completion_path = output / "_SUCCESS"
    owner = _load_json(owner_path)
    expected = _mapping(owner.get("files"), "output checksum manifest")
    actual = {
        path.relative_to(output).as_posix(): _sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path not in {owner_path, completion_path}
    }
    content_sha = hashlib.sha256(_canonical_json(actual)).hexdigest()
    if (
        dict(expected) != actual
        or owner.get("content_sha256") != content_sha
    ):
        raise SensitivityWorkflowError(
            "output content checksum verification failed"
        )
    return owner, content_sha


def aggregate_verified_shards(
    work_root: str | Path,
    output: str | Path,
) -> Mapping[str, Any]:
    """CPU-only strict aggregation under an exclusive analysis lock."""

    work = Path(work_root).resolve()
    with _ExclusiveLock(work / "locks" / "aggregate.lock"):
        return _aggregate_verified_shards_unlocked(work, Path(output))


def _aggregate_verified_shards_unlocked(
    work_root: str | Path,
    output: str | Path,
) -> Mapping[str, Any]:
    """CPU-only strict aggregation of the exact complete shard inventory."""

    work = Path(work_root).resolve()
    manifest = _load_analysis_manifest(work)
    analysis_sha = str(manifest.get("analysis_input_sha256", ""))
    pilot_path = _pilot_path(work)
    pilot = _verify_bound_json(
        pilot_path,
        analysis_input_sha256=analysis_sha,
    )
    _validate_pilot_payload(pilot, manifest=manifest)
    pilot_sha = _sha256_file(pilot_path)
    review_path = _identical_review_path(work)
    review_sha = _sha256_file(review_path)
    review = _validate_reviewed_identical(
        work,
        analysis_input_sha256=analysis_sha,
        reviewed_sha256=review_sha,
        reviewed_pilot_sha256=pilot_sha,
    )
    common = _mapping(manifest.get("common_identity"), "common identity")
    masks = common.get("whole_node_masks")
    if not isinstance(masks, Sequence) or len(masks) != 3:
        raise SensitivityWorkflowError(
            "analysis manifest lacks three whole-node masks"
        )
    mask_rows = [
        _mapping(row, "whole-node mask row") for row in masks
    ]
    mask_ids = [str(row["entry_id"]) for row in mask_rows]
    inventory: list[dict[str, object]] = []
    shard_paths: list[Path] = []
    trained_rows = {pair: [] for pair in _TRAINED_PAIRS}
    identical_rows: list[ProbeSufficientStatistics] = []
    randomized_rows = {
        seed: [] for seed in _RANDOM_SEEDS
    }
    common_probes: dict[int, bytes] = {}
    model_identities: dict[tuple[str, int | None], bytes] = {}
    driver_versions: set[str] = {
        str(
            _mapping(
                _mapping(pilot.get("resource"), "pilot resource").get(
                    "isolated_gpu"
                ),
                "pilot isolated GPU",
            ).get("driver_version")
        )
    }
    gpu_uuids_by_index: dict[int, set[str]] = {
        5: set(),
        6: set(),
        7: {
            str(
                _mapping(
                    _mapping(
                        pilot.get("resource"),
                        "pilot resource",
                    ).get("isolated_gpu"),
                    "pilot isolated GPU",
                ).get("physical_gpu_uuid")
            )
        },
    }
    expected_paths = [
        _shard_path(
            work,
            shard_type=shard_type,
            mask_replicate=mask_index,
            control_seed=seed,
        )
        for mask_index in range(3)
        for shard_type, seed in (
            ("trained", None),
            ("identical", None),
            *((("random", value) for value in _RANDOM_SEEDS)),
        )
    ]
    if len(expected_paths) != 30 or len(set(expected_paths)) != 30:
        raise SensitivityWorkflowError(
            "internal exact shard inventory is invalid"
        )
    _require_exact_bound_inventory(work / "shards", expected_paths)

    for mask_index, mask_row in enumerate(mask_rows):
        expected_probes = _expected_probe_records(
            mask_row,
            common=common,
        )
        for shard_type, seed in (
            ("trained", None),
            ("identical", None),
            *((("random", value) for value in _RANDOM_SEEDS)),
        ):
            path = _shard_path(
                work,
                shard_type=shard_type,
                mask_replicate=mask_index,
                control_seed=seed,
            )
            payload = _verify_bound_json(
                path,
                analysis_input_sha256=analysis_sha,
            )
            rows, model_identity = _validate_shard(
                payload,
                manifest=manifest,
                analysis_input_sha256=analysis_sha,
                reviewed_resource_pilot_sha256=pilot_sha,
                reviewed_identical_control_sha256=(
                    None if shard_type == "identical" else review_sha
                ),
                shard_type=shard_type,
                mask_row=mask_row,
                control_seed=seed,
                expected_probe_records=expected_probes,
            )
            model_key = (shard_type, seed)
            if model_key not in model_identities:
                model_identities[model_key] = model_identity
            elif model_identities[model_key] != model_identity:
                raise SensitivityWorkflowError(
                    "model state identity changed across mask shards"
                )
            probes = _canonical_json(payload["probe_records"])
            if mask_index not in common_probes:
                common_probes[mask_index] = probes
            elif common_probes[mask_index] != probes:
                raise SensitivityWorkflowError(
                    "shards did not use identical common probes"
                )
            if shard_type == "trained":
                for pair in _TRAINED_PAIRS:
                    trained_rows[pair].extend(rows[pair])
            elif shard_type == "identical":
                identical_rows.extend(rows[_IDENTICAL_PAIR])
            else:
                assert seed is not None
                pair = _expected_pairs("random", seed)[0]
                randomized_rows[seed].extend(rows[pair])
            shard_paths.append(path)
            resource = _mapping(
                payload.get("resource"),
                "shard resource",
            )
            isolated_gpu = _mapping(
                resource.get("isolated_gpu"),
                "shard isolated GPU",
            )
            driver_versions.add(str(isolated_gpu.get("driver_version")))
            physical_index = _integer(
                isolated_gpu.get("physical_gpu_index"),
                "shard physical GPU index",
            )
            gpu_uuids_by_index[physical_index].add(
                str(isolated_gpu.get("physical_gpu_uuid"))
            )
            inventory.append(
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "shard_type": shard_type,
                    "mask_replicate": mask_index,
                    "mask_entry_id": mask_row["entry_id"],
                    "control_seed": seed,
                    "full_graph_vjp_count": int(
                        resource.get("full_graph_vjp_count", -1)
                    ),
                    "physical_gpu_uuid": isolated_gpu[
                        "physical_gpu_uuid"
                    ],
                    "driver_version": isolated_gpu["driver_version"],
                }
            )

    if (
        len(shard_paths) != 30
        or sum(
            int(row["full_graph_vjp_count"]) for row in inventory
        )
        != 2_304
        or len(model_identities) != 10
        or len(driver_versions) != 1
        or any(
            len(gpu_uuids_by_index[index]) != 1
            for index in (5, 6, 7)
        )
    ):
        raise SensitivityWorkflowError(
            "aggregate inventory/model/driver contract is incomplete"
        )

    primary = {
        seed: _estimate_rows(
            trained_rows[(f"current_s{seed}", f"wider_s{seed}")],
            pair=(f"current_s{seed}", f"wider_s{seed}"),
            mask_ids=mask_ids,
        )
        for seed in range(3)
    }
    current_pairs = (
        ("current_s0", "current_s1"),
        ("current_s0", "current_s2"),
        ("current_s1", "current_s2"),
    )
    wider_pairs = (
        ("wider_s0", "wider_s1"),
        ("wider_s0", "wider_s2"),
        ("wider_s1", "wider_s2"),
    )
    within_current = {
        _pair_key(pair): _estimate_rows(
            trained_rows[pair],
            pair=pair,
            mask_ids=mask_ids,
        )
        for pair in current_pairs
    }
    within_wider = {
        _pair_key(pair): _estimate_rows(
            trained_rows[pair],
            pair=pair,
            mask_ids=mask_ids,
        )
        for pair in wider_pairs
    }
    identical = _estimate_rows(
        identical_rows,
        pair=_IDENTICAL_PAIR,
        mask_ids=mask_ids,
    )
    randomized = {}
    for seed in _RANDOM_SEEDS:
        pair = _expected_pairs("random", seed)[0]
        randomized[seed] = _estimate_rows(
            randomized_rows[seed],
            pair=pair,
            mask_ids=mask_ids,
        )
    decision = evaluate_operational_match(
        primary_by_seed=primary,
        within_current=within_current,
        within_wider=within_wider,
        identical=identical,
        randomized_by_seed=randomized,
    )
    estimates: dict[str, Mapping[str, PairEstimate] | PairEstimate] = {
        "primary": {str(seed): value for seed, value in primary.items()},
        "within_current": within_current,
        "within_wider": within_wider,
        "identical": identical,
        "randomized": {
            str(seed): value for seed, value in randomized.items()
        },
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": (
            "g2_token_categorical_sensitivity_post_hoc_comparison"
        ),
        "protocol": PROTOCOL_VERSION,
        "analysis_mode": _ANALYSIS_MODE,
        "analysis_input_sha256": analysis_sha,
        "analysis_manifest_sha256": _sha256_file(
            work / "analysis_manifest.json"
        ),
        "resource_pilot": {
            "path": str(pilot_path),
            "sha256": pilot_sha,
            "passes": True,
        },
        "identical_control_review": {
            "path": str(review_path),
            "sha256": review_sha,
            "passes": review["passes"],
        },
        "status": (
            "post_hoc_operational_match"
            if decision["operational_match"]
            else (
                "post_hoc_operational_mismatch"
                if decision["analysis_numerically_valid"]
                else "post_hoc_numerically_invalid"
            )
        ),
        "prespecified_gate": manifest["prespecified_gate"],
        "interval_interpretation": (
            "95% technical mask-plus-Rademacher-trace-probe bootstrap "
            "variability only; not biological, patient, or training-seed "
            "uncertainty"
        ),
        "shard_inventory": inventory,
        "verified_shard_count": len(inventory),
        "verified_full_graph_vjp_count": sum(
            int(row["full_graph_vjp_count"]) for row in inventory
        ),
        "common_probe_manifest_sha256": hashlib.sha256(
            b"".join(common_probes[index] for index in range(3))
        ).hexdigest(),
        "estimates": {
            "primary": {
                str(seed): value.to_dict()
                for seed, value in primary.items()
            },
            "within_current": {
                label: value.to_dict()
                for label, value in within_current.items()
            },
            "within_wider": {
                label: value.to_dict()
                for label, value in within_wider.items()
            },
            "identical": identical.to_dict(),
            "randomized": {
                str(seed): value.to_dict()
                for seed, value in randomized.items()
            },
        },
        "decision": decision,
        "limitations": [
            (
                "The prespecified 95-percent gate failed; this analysis is "
                "post-hoc exploratory and cannot revise that gate."
            ),
            (
                "Token prediction was dominated by the zero class and did "
                "not exceed the modal baseline on balanced/nonzero accuracy."
            ),
            (
                "Local gradients on one transductive core do not establish "
                "patient generalization, biological mechanism, or causality."
            ),
            (
                "Paired random controls can share RNG prefixes where "
                "parameter shapes align, conservatively raising similarity."
            ),
        ],
    }
    pair_rows = _pair_estimate_rows(estimates)
    _write_output(
        Path(output).resolve(),
        result=result,
        pair_rows=pair_rows,
        shard_paths=shard_paths,
        analysis_manifest_path=work / "analysis_manifest.json",
        pilot_path=pilot_path,
        identical_review_path=review_path,
    )
    return result


def _retain_labels(command: str) -> set[str]:
    if command == "pilot":
        return {"wider_s0"}
    if command == "trained-shard":
        return {
            *(f"current_s{seed}" for seed in range(3)),
            *(f"wider_s{seed}" for seed in range(3)),
        }
    if command == "identical-shard":
        return {"current_s0"}
    return set()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument(
        "--current-run",
        action="append",
        required=True,
        type=Path,
        help="Current-width run, repeated in seed 0/1/2 order.",
    )
    parser.add_argument(
        "--wider-run",
        action="append",
        required=True,
        type=Path,
        help="Wider run, repeated in seed 0/1/2 order.",
    )
    parser.add_argument(
        "--active-analysis-dir",
        "--work-root",
        dest="active_analysis_dir",
        required=True,
        type=Path,
        help=(
            "Exclusive scratch directory for this immutable post-hoc "
            "analysis (the --work-root spelling is a compatibility alias)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--device", required=True)
    pilot.add_argument(
        "--expected-physical-gpu-uuid",
        required=True,
        help="Physical UUID assigned by the supervisor worker.",
    )

    identical = subparsers.add_parser("identical-shard")
    identical.add_argument("--mask-replicate", required=True, type=int)
    identical.add_argument("--device", required=True)
    identical.add_argument(
        "--expected-physical-gpu-uuid",
        required=True,
        help="Physical UUID assigned by the supervisor worker.",
    )
    identical.add_argument("--reviewed-pilot-sha256", required=True)

    review = subparsers.add_parser("review-identical")
    review.add_argument("--reviewed-pilot-sha256", required=True)

    trained = subparsers.add_parser("trained-shard")
    trained.add_argument("--mask-replicate", required=True, type=int)
    trained.add_argument("--device", required=True)
    trained.add_argument(
        "--expected-physical-gpu-uuid",
        required=True,
        help="Physical UUID assigned by the supervisor worker.",
    )
    trained.add_argument("--reviewed-pilot-sha256", required=True)
    trained.add_argument(
        "--reviewed-identical-sha256",
        required=True,
    )

    random = subparsers.add_parser("random-shard")
    random.add_argument("--mask-replicate", required=True, type=int)
    random.add_argument(
        "--control-seed",
        required=True,
        type=int,
        choices=_RANDOM_SEEDS,
    )
    random.add_argument("--device", required=True)
    random.add_argument(
        "--expected-physical-gpu-uuid",
        required=True,
        help="Physical UUID assigned by the supervisor worker.",
    )
    random.add_argument("--reviewed-pilot-sha256", required=True)
    random.add_argument(
        "--reviewed-identical-sha256",
        required=True,
    )

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command in {
        "pilot",
        "trained-shard",
        "identical-shard",
        "random-shard",
    }:
        # This must precede any call that could initialize CUDA.
        _enforce_cublas_workspace_config()
    retain = _retain_labels(arguments.command)
    runs = _load_runs(
        arguments.current_run,
        arguments.wider_run,
        retain_labels=retain,
    )
    comparison_path = arguments.comparison.resolve()
    comparison = _validate_failed_gate_comparison(
        comparison_path,
        runs,
    )
    manifest_core = _analysis_manifest_core(
        comparison_path=comparison_path,
        comparison=comparison,
        runs=runs,
    )
    _, analysis_sha = _initialize_work_root(
        arguments.active_analysis_dir.resolve(),
        manifest_core,
    )
    if arguments.command == "aggregate":
        result = aggregate_verified_shards(
            arguments.active_analysis_dir,
            arguments.output,
        )
    elif arguments.command == "review-identical":
        result = review_identical_shards(
            arguments.active_analysis_dir,
            reviewed_pilot_sha256=arguments.reviewed_pilot_sha256,
        )
        result = {
            **dict(result),
            "identical_control_review_path": str(
                _identical_review_path(
                    arguments.active_analysis_dir.resolve()
                )
            ),
            "identical_control_review_sha256": _sha256_file(
                _identical_review_path(
                    arguments.active_analysis_dir.resolve()
                )
            ),
        }
    else:
        device, gpu_identity = _configure_device(
            arguments.device,
            expected_physical_gpu_uuid=(
                arguments.expected_physical_gpu_uuid
            ),
        )
        expected_gpu_index = (
            7
            if arguments.command == "pilot"
            else 5 + int(arguments.mask_replicate)
        )
        if gpu_identity.get("physical_gpu_index") != expected_gpu_index:
            raise SensitivityWorkflowError(
                f"{arguments.command} is assigned to physical GPU "
                f"{expected_gpu_index}, not "
                f"{gpu_identity.get('physical_gpu_index')}"
            )
        if arguments.command == "pilot":
            result = _run_pilot(
                work_root=arguments.active_analysis_dir.resolve(),
                analysis_input_sha256=analysis_sha,
                runs=runs,
                device=device,
                gpu_identity=gpu_identity,
            )
            result = {
                **dict(result),
                "resource_pilot_path": str(
                    _pilot_path(arguments.active_analysis_dir.resolve())
                ),
                "resource_pilot_sha256": _sha256_file(
                    _pilot_path(arguments.active_analysis_dir.resolve())
                ),
            }
        else:
            shard_type = arguments.command.removesuffix("-shard")
            control_seed = (
                arguments.control_seed
                if shard_type == "random"
                else None
            )
            result = _run_shard(
                work_root=arguments.active_analysis_dir.resolve(),
                analysis_input_sha256=analysis_sha,
                runs=runs,
                shard_type=shard_type,
                mask_replicate=arguments.mask_replicate,
                control_seed=control_seed,
                reviewed_pilot_sha256=(
                    arguments.reviewed_pilot_sha256
                ),
                reviewed_identical_sha256=getattr(
                    arguments,
                    "reviewed_identical_sha256",
                    None,
                ),
                device=device,
                gpu_identity=gpu_identity,
            )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SensitivityWorkflowError as error:
        print(f"Sensitivity workflow failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error

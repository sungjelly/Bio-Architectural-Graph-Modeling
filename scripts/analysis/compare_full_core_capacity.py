#!/usr/bin/env python3
"""Compare the locked full-core G2 and G2-matched self-only runs.

This is a strict post-run analysis. It accepts only two checksum-verified,
successful, conclusion-bearing, 200-epoch bundles produced by the held-in
full-core protocol. A completed comparison is a scientific result even when
the locked gate fails, so a negative gate does not make the command fail.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.run_archive import (  # noqa: E402
    RunValidationError,
    verify_run_bundle,
)


_PROTOCOL = "held_in_full_core_fixed_budget"
_CAMPAIGN_ID = "cmp_20260725_full_core_high_k_capacity"
_GRAPH_EXECUTION = "full_core_exact_no_neighbor_sampling"
_CHECKPOINT_POLICY = "final_epoch_no_validation_selection"
_PRIMARY_METRIC = "fit/whole_node/masked_huber"
_EXPECTED_EPOCHS = 200
_EXPECTED_REPLICATES = 3
_TABLE_SUFFIXES = (".parquet", ".jsonl", ".csv")
_G2_KEYS = {"g2"}
_SELF_KEYS = {"b0g2matched"}
_PAIR_DATA_FIELDS = (
    "dataset_id",
    "version",
    "dataset_fingerprint",
    "preprocessing_version",
    "split_id",
    "split_fingerprint",
)
_FORBIDDEN_METRIC_PREFIXES = ("val/", "validation/", "test/", "external/")
_MEAN_REL_TOL = 1e-9
_MEAN_ABS_TOL = 1e-12


class CapacityComparisonError(RuntimeError):
    """Raised when inputs do not satisfy the locked comparison contract."""


@dataclass(frozen=True)
class RunEvidence:
    role: str
    root: Path
    run_id: str
    model_name: str
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    final_metrics: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    replicates: tuple[Mapping[str, Any], ...]
    convergence: Mapping[str, Any]
    training_provenance: Mapping[str, Any]
    full_core_inputs: Mapping[str, Any]
    fixed_masks: Mapping[str, Any]
    data_provenance: Mapping[str, Any]
    split_provenance: Mapping[str, Any]


def _model_key(value: object) -> str:
    return "".join(
        character
        for character in str(value).strip().lower()
        if character.isalnum()
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapacityComparisonError(f"{label} must be a mapping")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CapacityComparisonError(f"cannot read JSON mapping: {path}") from error
    return _mapping(value, path.as_posix())


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CapacityComparisonError(f"cannot read YAML mapping: {path}") from error
    return _mapping(value, path.as_posix())


def _logical_table_path(root: Path, stem: str) -> Path:
    matches = [
        root / f"{stem}{suffix}"
        for suffix in _TABLE_SUFFIXES
        if (root / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise CapacityComparisonError(
            f"{root} requires exactly one {stem} table; found "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def _load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    rows.append(dict(_mapping(value, f"{path}:{line_number}")))
        except (OSError, ValueError) as error:
            raise CapacityComparisonError(f"cannot read JSONL table: {path}") from error
    elif path.suffix == ".csv":
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except (OSError, csv.Error) as error:
            raise CapacityComparisonError(f"cannot read CSV table: {path}") from error
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except (ImportError, ModuleNotFoundError) as error:
            raise CapacityComparisonError(
                f"reading Parquet requires pyarrow: {path}"
            ) from error
        try:
            rows = [
                dict(_mapping(row, f"row in {path}"))
                for row in parquet.read_table(path).to_pylist()
            ]
        except Exception as error:
            raise CapacityComparisonError(f"cannot read Parquet table: {path}") from error
    else:
        raise CapacityComparisonError(f"unsupported table format: {path}")
    if not rows:
        raise CapacityComparisonError(f"table is empty: {path}")
    return rows


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CapacityComparisonError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise CapacityComparisonError(f"{label} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise CapacityComparisonError(f"{label} must be an integer")
    return converted


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CapacityComparisonError(f"{label} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise CapacityComparisonError(f"{label} must be numeric") from error


def _finite_float(value: Any, label: str) -> float:
    converted = _as_float(value, label)
    if not math.isfinite(converted):
        raise CapacityComparisonError(f"{label} must be finite")
    return converted


def _same_number(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_MEAN_REL_TOL,
        abs_tol=_MEAN_ABS_TOL,
    )


def _sha256(value: Any, label: str) -> str:
    checksum = str(value)
    if len(checksum) != 64 or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        raise CapacityComparisonError(f"{label} must be a lowercase SHA-256")
    return checksum


def _assert_no_held_out_artifacts(
    root: Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    replicates: Sequence[Mapping[str, Any]],
) -> None:
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    masking = _mapping(config.get("masking"), "config.masking")
    dataset = _mapping(config.get("dataset"), "config.dataset")
    if list(evaluation.get("splits", ())) != ["fit"]:
        raise CapacityComparisonError(f"{root} does not declare fit as its only role")
    if evaluation.get("canonical_prediction_split") != "fit":
        raise CapacityComparisonError(f"{root} canonical prediction role is not fit")
    if summary.get("canonical_prediction_split") != "fit":
        raise CapacityComparisonError(f"{root} summary prediction role is not fit")
    if summary.get("generalization_estimate") is not False:
        raise CapacityComparisonError(
            f"{root} must explicitly declare generalization_estimate=false"
        )
    if evaluation.get("generalization_estimate") is not False:
        raise CapacityComparisonError(
            f"{root} evaluation does not declare generalization_estimate=false"
        )
    if evaluation.get("validation_or_test_selection") is not False:
        raise CapacityComparisonError(
            f"{root} evaluation permits validation/test selection"
        )
    if dataset.get("validation_or_test_partition_present") is not False:
        raise CapacityComparisonError(
            f"{root} dataset does not exclude validation/test partitions"
        )
    if dataset.get("experimental_unit") != "single_spatial_core":
        raise CapacityComparisonError(
            f"{root} is not declared as a single-spatial-core run"
        )
    for role in ("validation", "test"):
        if _as_int(
            masking.get(f"{role}_replicates"),
            f"{root} masking.{role}_replicates",
        ) != 0:
            raise CapacityComparisonError(
                f"{root} configures {role} mask replicates"
            )

    forbidden_files = sorted(
        path.relative_to(root).as_posix()
        for split in ("validation", "test", "external")
        for path in (root / "predictions").glob(f"{split}.*")
    )
    if forbidden_files:
        raise CapacityComparisonError(
            f"{root} contains held-out prediction artifacts: {forbidden_files}"
        )

    metric_mappings: list[tuple[str, Mapping[str, Any]]] = [
        ("metrics/final.json", final_metrics)
    ]
    summary_metrics = summary.get("metrics")
    if summary_metrics is not None:
        metric_mappings.append(
            ("summary.metrics", _mapping(summary_metrics, "summary.metrics"))
        )
    for source, values in metric_mappings:
        forbidden = sorted(
            str(name)
            for name in values
            if str(name).startswith(_FORBIDDEN_METRIC_PREFIXES)
        )
        if forbidden:
            raise CapacityComparisonError(
                f"{root} contains held-out metrics in {source}: {forbidden}"
            )

    events = root / "metrics/events.jsonl"
    try:
        event_rows = [
            json.loads(line)
            for line in events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as error:
        raise CapacityComparisonError(f"cannot inspect metric events: {events}") from error
    forbidden_events = sorted(
        str(event.get("name"))
        for event in event_rows
        if str(event.get("name", "")).startswith(_FORBIDDEN_METRIC_PREFIXES)
    )
    if forbidden_events:
        raise CapacityComparisonError(
            f"{root} contains held-out metric events: {forbidden_events}"
        )
    if any(str(row.get("split")) != "fit" for row in history):
        raise CapacityComparisonError(f"{root} history contains a non-fit role")
    if any(str(row.get("split")) != "fit" for row in replicates):
        raise CapacityComparisonError(
            f"{root} evaluation replicates contain a non-fit role"
        )
    for table_name, rows in (
        ("history", history),
        ("evaluation replicates", replicates),
    ):
        forbidden_columns = sorted(
            {
                str(column)
                for row in rows
                for column in row
                if str(column).startswith(_FORBIDDEN_METRIC_PREFIXES)
            }
        )
        if forbidden_columns:
            raise CapacityComparisonError(
                f"{root} {table_name} contains held-out columns: "
                f"{forbidden_columns}"
            )


def _validate_epoch_structure(
    root: Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    convergence: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
) -> None:
    trainer = _mapping(config.get("trainer"), "config.trainer")
    expected_values = {
        "config trainer max_epochs": trainer.get("max_epochs"),
        "summary fixed_epoch_budget": summary.get("fixed_epoch_budget"),
        "training provenance fixed_epoch_budget": training_provenance.get(
            "fixed_epoch_budget"
        ),
    }
    for label, value in expected_values.items():
        if _as_int(value, f"{root} {label}") != _EXPECTED_EPOCHS:
            raise CapacityComparisonError(f"{root} {label} is not 200")
    if trainer.get("fixed_epoch_budget") is not True:
        raise CapacityComparisonError(f"{root} is not a fixed-budget run")
    if trainer.get("restore_best") is not False:
        raise CapacityComparisonError(f"{root} restored a selected checkpoint")
    if trainer.get("primary_checkpoint_role") != "last":
        raise CapacityComparisonError(f"{root} did not retain the final checkpoint")
    if trainer.get("neighbor_sampling") is not False:
        raise CapacityComparisonError(f"{root} used neighbor sampling")
    if trainer.get("graph_execution") != _GRAPH_EXECUTION:
        raise CapacityComparisonError(f"{root} used the wrong graph execution")
    if summary.get("checkpoint_role") != "last":
        raise CapacityComparisonError(f"{root} summary checkpoint role is not last")
    if training_provenance.get("training_protocol") != _PROTOCOL:
        raise CapacityComparisonError(
            f"{root} training provenance uses the wrong protocol"
        )
    if training_provenance.get("graph_execution") != _GRAPH_EXECUTION:
        raise CapacityComparisonError(
            f"{root} training provenance uses the wrong graph execution"
        )
    if training_provenance.get("checkpoint_policy") != _CHECKPOINT_POLICY:
        raise CapacityComparisonError(
            f"{root} training provenance uses the wrong checkpoint policy"
        )
    for label, value in (
        ("summary final_epoch", summary.get("final_epoch")),
        ("training provenance final_epoch", training_provenance.get("final_epoch")),
        ("convergence final_epoch", convergence.get("final_epoch")),
    ):
        if _as_int(value, f"{root} {label}") != _EXPECTED_EPOCHS - 1:
            raise CapacityComparisonError(f"{root} {label} is not 199")

    epochs = [_as_int(row.get("epoch"), f"{root} history epoch") for row in history]
    if epochs != list(range(_EXPECTED_EPOCHS)):
        raise CapacityComparisonError(
            f"{root} history must contain epochs 0..199 exactly once in order"
        )
    if any(row.get("training_protocol") != _PROTOCOL for row in history):
        raise CapacityComparisonError(
            f"{root} history contains the wrong training protocol"
        )
    if convergence.get("all_epochs_completed") is not True:
        raise CapacityComparisonError(
            f"{root} convergence diagnostic does not confirm all epochs"
        )


def _validate_locked_scope(
    root: Path,
    *,
    config: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
    full_core_inputs: Mapping[str, Any],
    fixed_masks: Mapping[str, Any],
) -> None:
    dataset = _mapping(config.get("dataset"), "config.dataset")
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    masking = _mapping(config.get("masking"), "config.masking")
    graph = _mapping(config.get("graph"), "config.graph")
    campaign = _mapping(config.get("campaign"), "config.campaign")
    for field in _PAIR_DATA_FIELDS:
        value = dataset.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise CapacityComparisonError(
                f"{root} config.dataset.{field} is missing"
            )
    dataset_checksum = _sha256(
        dataset.get("dataset_fingerprint"),
        f"{root} config.dataset.dataset_fingerprint",
    )
    split_checksum = _sha256(
        dataset.get("split_fingerprint"),
        f"{root} config.dataset.split_fingerprint",
    )
    if data_provenance.get("dataset_fingerprint") != dataset_checksum:
        raise CapacityComparisonError(
            f"{root} dataset fingerprint disagrees with provenance"
        )
    if data_provenance.get("preprocessing_version") != dataset.get(
        "preprocessing_version"
    ):
        raise CapacityComparisonError(
            f"{root} preprocessing version disagrees with provenance"
        )
    if split_provenance.get("split_fingerprint") != split_checksum:
        raise CapacityComparisonError(
            f"{root} split fingerprint disagrees with provenance"
        )
    preprocessing = _mapping(
        full_core_inputs.get("preprocessing_checksums"),
        f"{root} full_core_inputs.preprocessing_checksums",
    )
    materialized = _mapping(
        full_core_inputs.get("materialized_identity_verification"),
        f"{root} full_core_inputs.materialized_identity_verification",
    )
    if preprocessing.get("preprocessing_sha256") != dataset_checksum:
        raise CapacityComparisonError(
            f"{root} materialized preprocessing checksum drifted"
        )
    if (
        materialized.get("dataset_fingerprint") != dataset_checksum
        or materialized.get("split_fingerprint") != split_checksum
    ):
        raise CapacityComparisonError(
            f"{root} materialized dataset/split identity drifted"
        )

    if campaign.get("campaign_id") != _CAMPAIGN_ID:
        raise CapacityComparisonError(f"{root} belongs to the wrong campaign")
    if graph.get("kind") != "exact_spatial_knn_radius_guard":
        raise CapacityComparisonError(f"{root} is not the locked exact graph")
    for field in ("k", "neighbor_k"):
        if _as_int(graph.get(field), f"{root} graph.{field}") != 1000:
            raise CapacityComparisonError(f"{root} graph.{field} is not 1000")
    if graph.get("full_core_graph") is not True:
        raise CapacityComparisonError(f"{root} does not use the full-core graph")
    if graph.get("symmetry") != "mutual":
        raise CapacityComparisonError(f"{root} graph is not mutual")
    if _finite_float(graph.get("edge_dropout"), f"{root} graph.edge_dropout") != 0:
        raise CapacityComparisonError(f"{root} graph edge dropout is not zero")
    if _as_int(
        evaluation.get("mask_replicates_per_mode"),
        f"{root} evaluation.mask_replicates_per_mode",
    ) != _EXPECTED_REPLICATES:
        raise CapacityComparisonError(
            f"{root} evaluation does not configure three mask replicates"
        )
    if _as_int(
        masking.get("fit_replicates"),
        f"{root} masking.fit_replicates",
    ) != _EXPECTED_REPLICATES:
        raise CapacityComparisonError(
            f"{root} masking does not configure three fit replicates"
        )
    if evaluation.get("fixed_mask_bundle") is not True:
        raise CapacityComparisonError(
            f"{root} evaluation does not require a fixed mask bundle"
        )
    if fixed_masks.get("used_for_gradient_updates") is not False:
        raise CapacityComparisonError(
            f"{root} evaluation masks were not excluded from gradient updates"
        )
    if fixed_masks.get("used_for_checkpoint_selection") is not False:
        raise CapacityComparisonError(
            f"{root} evaluation masks were not excluded from checkpoint selection"
        )


def _load_run(path: str | Path, *, role: str, expected_models: set[str]) -> RunEvidence:
    root = Path(path).resolve(strict=False)
    try:
        verification = verify_run_bundle(root)
    except (RunValidationError, OSError, ValueError) as error:
        raise CapacityComparisonError(
            f"{role} bundle failed canonical verification: {root}"
        ) from error
    if verification.get("status") != "success":
        raise CapacityComparisonError(f"{role} bundle is not successful: {root}")

    config = _load_yaml(root / "config.resolved.yaml")
    summary = _load_json(root / "summary.json")
    final_metrics = _load_json(root / "metrics/final.json")
    convergence = _load_json(root / "diagnostics/training_convergence.json")
    training_provenance = _load_json(root / "provenance/full_core_training.json")
    full_core_inputs = _load_json(root / "provenance/full_core_inputs.json")
    fixed_masks = _load_json(root / "provenance/fixed_evaluation_masks.json")
    data_provenance = _load_json(root / "provenance/data_fingerprints.json")
    split_provenance = _load_json(root / "provenance/split_fingerprint.json")
    history = tuple(
        _load_table(_logical_table_path(root, "metrics/history"))
    )
    replicates = tuple(
        _load_table(_logical_table_path(root, "metrics/evaluation_replicates"))
    )

    if summary.get("status") != "success":
        raise CapacityComparisonError(f"{role} summary status is not success")
    if summary.get("training_exit_status") != "success":
        raise CapacityComparisonError(f"{role} training exit status is not success")
    run_id = str(summary.get("run_id", ""))
    if run_id != root.name:
        raise CapacityComparisonError(f"{role} summary run_id does not match path")
    model = _mapping(config.get("model"), "config.model")
    model_name = str(model.get("name", ""))
    if _model_key(model_name) not in expected_models:
        raise CapacityComparisonError(
            f"{role} model {model_name!r} is not the required model family"
        )
    if str(summary.get("model_name")) != model_name:
        raise CapacityComparisonError(f"{role} summary model identity drifted")

    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    if evaluation.get("protocol") != _PROTOCOL:
        raise CapacityComparisonError(f"{role} uses the wrong evaluation protocol")
    if summary.get("evaluation_protocol") != _PROTOCOL:
        raise CapacityComparisonError(f"{role} summary uses the wrong protocol")
    if summary.get("diagnostic_resource_pilot") is not False:
        raise CapacityComparisonError(f"{role} is a diagnostic resource pilot")
    if summary.get("conclusion_eligible") is not True:
        raise CapacityComparisonError(f"{role} is not conclusion-eligible")
    if (
        summary.get(
            "evaluation_metrics_include_all_configured_replicates_per_mode"
        )
        is not True
    ):
        raise CapacityComparisonError(
            f"{role} does not confirm complete evaluation replicates"
        )
    if _as_int(
        summary.get("evaluation_mask_replicates_per_mode"),
        f"{role} evaluation replicate count",
    ) != _EXPECTED_REPLICATES:
        raise CapacityComparisonError(f"{role} does not declare three replicates")
    canonical_prediction = _mapping(
        summary.get("canonical_prediction_selection"),
        f"{role} summary canonical_prediction_selection",
    )
    if (
        canonical_prediction.get("split") != "fit"
        or canonical_prediction.get("mask_mode") != "whole_node"
        or _as_int(
            canonical_prediction.get("mask_replicate"),
            f"{role} canonical prediction replicate",
        )
        != 0
    ):
        raise CapacityComparisonError(
            f"{role} canonical prediction selection is not fit/whole_node/0"
        )

    _validate_locked_scope(
        root,
        config=config,
        data_provenance=data_provenance,
        split_provenance=split_provenance,
        full_core_inputs=full_core_inputs,
        fixed_masks=fixed_masks,
    )
    _assert_no_held_out_artifacts(
        root,
        config=config,
        summary=summary,
        final_metrics=final_metrics,
        history=history,
        replicates=replicates,
    )
    _validate_epoch_structure(
        root,
        config=config,
        summary=summary,
        history=history,
        convergence=convergence,
        training_provenance=training_provenance,
    )
    return RunEvidence(
        role=role,
        root=root,
        run_id=run_id,
        model_name=model_name,
        config=config,
        summary=summary,
        final_metrics=final_metrics,
        history=history,
        replicates=replicates,
        convergence=convergence,
        training_provenance=training_provenance,
        full_core_inputs=full_core_inputs,
        fixed_masks=fixed_masks,
        data_provenance=data_provenance,
        split_provenance=split_provenance,
    )


def _canonical_subset(mapping: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: mapping.get(field) for field in fields}


def _parameter_count(run: RunEvidence) -> int:
    sources = {
        "summary": run.summary.get("parameter_count"),
        "metrics/final": run.final_metrics.get("resource/parameter_count"),
        "training provenance": run.training_provenance.get("parameter_count"),
    }
    converted = {
        source: _as_int(value, f"{run.role} {source} parameter_count")
        for source, value in sources.items()
    }
    if len(set(converted.values())) != 1:
        raise CapacityComparisonError(
            f"{run.role} parameter counts disagree: {converted}"
        )
    result = next(iter(converted.values()))
    if result <= 0:
        raise CapacityComparisonError(f"{run.role} parameter count must be positive")
    return result


def _paired_epoch_masks(
    g2: RunEvidence, matched_self: RunEvidence
) -> None:
    string_fields = ("mask_mode", "mask_checksum")
    integer_fields = (
        "epoch",
        "mask_seed",
        "edge_dropout_seed",
        "n_masked_entries",
        "n_target_nodes",
    )
    for row_index, (g2_row, self_row) in enumerate(
        zip(g2.history, matched_self.history, strict=True)
    ):
        g2_key = tuple(str(g2_row.get(field, "")) for field in string_fields) + tuple(
            _as_int(
                g2_row.get(field),
                f"G2 history row {row_index} {field}",
            )
            for field in integer_fields
        )
        self_key = tuple(
            str(self_row.get(field, "")) for field in string_fields
        ) + tuple(
            _as_int(
                self_row.get(field),
                f"matched-self history row {row_index} {field}",
            )
            for field in integer_fields
        )
        if g2_key != self_key:
            raise CapacityComparisonError(
                "epoch training masks are not paired at epoch "
                f"{g2_row.get('epoch')}"
            )


def _verify_pair(g2: RunEvidence, matched_self: RunEvidence) -> int:
    if g2.run_id == matched_self.run_id:
        raise CapacityComparisonError("G2 and matched-self inputs are the same run")
    g2_dataset = _mapping(g2.config.get("dataset"), "g2 config.dataset")
    self_dataset = _mapping(
        matched_self.config.get("dataset"), "matched-self config.dataset"
    )
    if _canonical_subset(g2_dataset, _PAIR_DATA_FIELDS) != _canonical_subset(
        self_dataset, _PAIR_DATA_FIELDS
    ):
        raise CapacityComparisonError("runs do not share dataset and split identity")
    g2_campaign = _mapping(g2.config.get("campaign"), "g2 config.campaign")
    self_campaign = _mapping(
        matched_self.config.get("campaign"), "matched-self config.campaign"
    )
    if g2_campaign.get("campaign_id") != self_campaign.get("campaign_id"):
        raise CapacityComparisonError("paired runs belong to different campaigns")
    if _as_int(g2.config.get("fold"), "G2 fold") != _as_int(
        matched_self.config.get("fold"), "matched-self fold"
    ):
        raise CapacityComparisonError("paired runs use different folds")
    for section in ("graph", "masking", "trainer", "evaluation"):
        if g2.config.get(section) != matched_self.config.get(section):
            raise CapacityComparisonError(f"paired runs differ in config.{section}")
    g2_features = _mapping(g2.config.get("features"), "g2 config.features")
    self_features = _mapping(
        matched_self.config.get("features"), "matched-self config.features"
    )
    node_feature_fields = (
        "fit_scope",
        "node_expression",
        "node_metadata",
        "prohibited_node_inputs",
    )
    if _canonical_subset(g2_features, node_feature_fields) != _canonical_subset(
        self_features, node_feature_fields
    ):
        raise CapacityComparisonError(
            "paired runs differ in their node-feature contract"
        )
    if _as_int(g2.config.get("seed"), "G2 seed") != _as_int(
        matched_self.config.get("seed"), "matched-self seed"
    ):
        raise CapacityComparisonError("paired runs use different model seeds")
    for run in (g2, matched_self):
        configured_seed = _as_int(run.config.get("seed"), f"{run.role} config seed")
        if _as_int(run.summary.get("model_seed"), f"{run.role} summary seed") != (
            configured_seed
        ):
            raise CapacityComparisonError(f"{run.role} model seed drifted")
        if _as_int(
            run.training_provenance.get("model_seed"),
            f"{run.role} training provenance seed",
        ) != configured_seed:
            raise CapacityComparisonError(f"{run.role} training seed drifted")

    for source in ("data_provenance", "split_provenance"):
        if getattr(g2, source) != getattr(matched_self, source):
            raise CapacityComparisonError(f"paired runs differ in {source}")
    g2_inputs = g2.full_core_inputs
    self_inputs = matched_self.full_core_inputs
    for field in ("preprocessing_checksums", "graph_checksums", "graph_config"):
        if g2_inputs.get(field) != self_inputs.get(field):
            raise CapacityComparisonError(
                f"paired runs differ in full-core {field}"
            )
    g2_graph_checksum = _sha256(
        g2.summary.get("graph_sha256"), "G2 summary graph_sha256"
    )
    self_graph_checksum = _sha256(
        matched_self.summary.get("graph_sha256"),
        "matched-self summary graph_sha256",
    )
    if g2_graph_checksum != self_graph_checksum:
        raise CapacityComparisonError("paired runs use different materialized graphs")
    g2_edge_count = _as_int(
        g2.summary.get("graph_directed_edges"), "G2 graph_directed_edges"
    )
    self_edge_count = _as_int(
        matched_self.summary.get("graph_directed_edges"),
        "matched-self graph_directed_edges",
    )
    if g2_edge_count <= 0 or g2_edge_count != self_edge_count:
        raise CapacityComparisonError("paired runs use different graph edge counts")
    graph_checksums = _mapping(
        g2_inputs.get("graph_checksums"), "full_core_inputs.graph_checksums"
    )
    if g2_graph_checksum != _sha256(
        graph_checksums.get("graph_sha256"),
        "full_core_inputs.graph_checksums.graph_sha256",
    ):
        raise CapacityComparisonError("summary graph checksum is inconsistent")

    g2_mask_checksum = _sha256(
        g2.summary.get("evaluation_mask_bundle_sha256"),
        "G2 summary evaluation_mask_bundle_sha256",
    )
    self_mask_checksum = _sha256(
        matched_self.summary.get("evaluation_mask_bundle_sha256"),
        "matched-self summary evaluation_mask_bundle_sha256",
    )
    if g2_mask_checksum != self_mask_checksum:
        raise CapacityComparisonError("paired runs use different mask bundles")
    if g2.fixed_masks.get("bundle_manifest") != matched_self.fixed_masks.get(
        "bundle_manifest"
    ):
        raise CapacityComparisonError("paired fixed-mask manifests differ")
    bundle = _mapping(
        g2.fixed_masks.get("bundle_manifest"), "fixed mask bundle manifest"
    )
    if _sha256(
        bundle.get("bundle_checksum"),
        "fixed mask bundle bundle_checksum",
    ) != g2_mask_checksum:
        raise CapacityComparisonError("summary mask checksum is inconsistent")
    _paired_epoch_masks(g2, matched_self)

    g2_count = _parameter_count(g2)
    self_count = _parameter_count(matched_self)
    if g2_count != self_count:
        raise CapacityComparisonError(
            f"parameter counts differ: G2={g2_count}, matched-self={self_count}"
        )
    return g2_count


def _normalized_replicates(run: RunEvidence) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    counts: dict[str, int] = {}
    for index, row in enumerate(run.replicates):
        split = str(row.get("split"))
        mode = str(row.get("mask_mode"))
        replicate = _as_int(
            row.get("mask_replicate"),
            f"{run.role} replicate row {index} mask_replicate",
        )
        checksum = str(row.get("mask_checksum", ""))
        n_masked = _as_int(
            row.get("n_masked"),
            f"{run.role} replicate row {index} n_masked",
        )
        huber = _finite_float(
            row.get("masked_huber"),
            f"{run.role} replicate row {index} masked_huber",
        )
        if split != "fit":
            raise CapacityComparisonError(
                f"{run.role} evaluation row {index} is not fit"
            )
        if mode not in {"partial_gene", "whole_node", "spatial_block"}:
            raise CapacityComparisonError(
                f"{run.role} evaluation row {index} has unknown mask mode"
            )
        if replicate not in range(_EXPECTED_REPLICATES):
            raise CapacityComparisonError(
                f"{run.role} evaluation row {index} has invalid replicate"
            )
        if len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise CapacityComparisonError(
                f"{run.role} evaluation row {index} has invalid mask checksum"
            )
        if huber < 0:
            raise CapacityComparisonError(
                f"{run.role} evaluation row {index} has negative Huber loss"
            )
        if n_masked <= 0:
            raise CapacityComparisonError(
                f"{run.role} evaluation row {index} has non-positive n_masked"
            )
        entry_id = str(row.get("mask_entry_id", ""))
        if not entry_id:
            raise CapacityComparisonError(
                f"{run.role} evaluation row {index} has no mask_entry_id"
            )
        identity = (mode, replicate)
        if identity in identities:
            raise CapacityComparisonError(
                f"{run.role} has duplicate evaluation row {identity}"
            )
        identities.add(identity)
        counts[mode] = counts.get(mode, 0) + 1
        rows.append(
            {
                "mask_mode": mode,
                "mask_replicate": replicate,
                "mask_checksum": checksum,
                "mask_entry_id": entry_id,
                "mask_seed": _as_int(
                    row.get("mask_seed"),
                    f"{run.role} evaluation row {index} mask_seed",
                ),
                "n_masked": n_masked,
                "masked_huber": huber,
            }
        )
    expected_counts = {
        "partial_gene": _EXPECTED_REPLICATES,
        "whole_node": _EXPECTED_REPLICATES,
        "spatial_block": _EXPECTED_REPLICATES,
    }
    if counts != expected_counts or len(rows) != 9:
        raise CapacityComparisonError(
            f"{run.role} evaluation table does not contain 3x3 rows: {counts}"
        )
    return rows


def _validate_replicates_against_manifest(
    run: RunEvidence,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    bundle = _mapping(
        run.fixed_masks.get("bundle_manifest"),
        f"{run.role} fixed mask bundle manifest",
    )
    raw_entries = bundle.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 9:
        raise CapacityComparisonError(
            f"{run.role} fixed-mask manifest does not contain 3x3 entries"
        )
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry, f"{run.role} mask manifest entry {index}")
        spec = _mapping(
            entry.get("spec"),
            f"{run.role} mask manifest entry {index} spec",
        )
        summary = _mapping(
            entry.get("summary"),
            f"{run.role} mask manifest entry {index} summary",
        )
        mode = str(spec.get("label", ""))
        replicate = _as_int(
            entry.get("replicate"),
            f"{run.role} mask manifest entry {index} replicate",
        )
        identity = (mode, replicate)
        if identity in entries:
            raise CapacityComparisonError(
                f"{run.role} fixed-mask manifest duplicates {identity}"
            )
        entries[identity] = {
            "mask_entry_id": str(entry.get("entry_id", "")),
            "mask_seed": _as_int(
                entry.get("seed"),
                f"{run.role} mask manifest entry {index} seed",
            ),
            "mask_checksum": _sha256(
                entry.get("mask_checksum"),
                f"{run.role} mask manifest entry {index} checksum",
            ),
            "n_masked": _as_int(
                summary.get("n_masked_entries"),
                f"{run.role} mask manifest entry {index} n_masked_entries",
            ),
            "split": str(entry.get("split", "")),
        }
    row_index = {
        (str(row["mask_mode"]), int(row["mask_replicate"])): row
        for row in rows
    }
    if set(entries) != set(row_index):
        raise CapacityComparisonError(
            f"{run.role} evaluation rows do not match fixed-mask manifest identities"
        )
    for identity, manifest_entry in entries.items():
        row = row_index[identity]
        observed = {
            field: row[field]
            for field in (
                "mask_entry_id",
                "mask_seed",
                "mask_checksum",
                "n_masked",
            )
        }
        expected = {
            field: manifest_entry[field]
            for field in (
                "mask_entry_id",
                "mask_seed",
                "mask_checksum",
                "n_masked",
            )
        }
        if manifest_entry["split"] != "fit" or observed != expected:
            raise CapacityComparisonError(
                f"{run.role} evaluation row {identity} does not match its "
                "fixed-mask manifest entry"
            )


def _reconcile_primary(run: RunEvidence, rows: Sequence[Mapping[str, Any]]) -> float:
    whole = [
        _finite_float(row["masked_huber"], f"{run.role} whole-node Huber")
        for row in rows
        if row["mask_mode"] == "whole_node"
    ]
    mean = statistics.fmean(whole)
    sources = {
        "metrics/final": run.final_metrics.get(_PRIMARY_METRIC),
        "summary primary_metric_value": run.summary.get("primary_metric_value"),
        "summary.metrics": _mapping(
            run.summary.get("metrics"), f"{run.role} summary.metrics"
        ).get(_PRIMARY_METRIC),
    }
    if run.summary.get("primary_metric_name") != _PRIMARY_METRIC:
        raise CapacityComparisonError(f"{run.role} primary metric name drifted")
    for source, value in sources.items():
        observed = _finite_float(value, f"{run.role} {source}")
        if not _same_number(mean, observed):
            raise CapacityComparisonError(
                f"{run.role} whole-node replicate mean {mean} does not match "
                f"{source} {observed}"
            )
    return mean


def _pair_whole_node_rows(
    g2_rows: Sequence[Mapping[str, Any]],
    self_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def index(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, int], Mapping[str, Any]]:
        selected = [row for row in rows if row["mask_mode"] == "whole_node"]
        return {
            (str(row["mask_checksum"]), int(row["mask_replicate"])): row
            for row in selected
        }

    g2_index = index(g2_rows)
    self_index = index(self_rows)
    if set(g2_index) != set(self_index) or len(g2_index) != _EXPECTED_REPLICATES:
        raise CapacityComparisonError(
            "whole-node rows are not paired by mask checksum and replicate"
        )
    paired: list[dict[str, Any]] = []
    for checksum, replicate in sorted(g2_index, key=lambda value: value[1]):
        g2_row = g2_index[(checksum, replicate)]
        self_row = self_index[(checksum, replicate)]
        if (
            g2_row["mask_entry_id"] != self_row["mask_entry_id"]
            or g2_row["mask_seed"] != self_row["mask_seed"]
            or g2_row["n_masked"] != self_row["n_masked"]
        ):
            raise CapacityComparisonError(
                f"whole-node mask provenance differs for replicate {replicate}"
            )
        g2_huber = float(g2_row["masked_huber"])
        self_huber = float(self_row["masked_huber"])
        difference = self_huber - g2_huber
        paired.append(
            {
                "mask_replicate": replicate,
                "mask_checksum": checksum,
                "g2_masked_huber": g2_huber,
                "matched_self_masked_huber": self_huber,
                "self_minus_g2_masked_huber": difference,
                "favors_g2": difference > 0.0,
            }
        )
    if [row["mask_replicate"] for row in paired] != [0, 1, 2]:
        raise CapacityComparisonError("whole-node replicates must be exactly 0, 1, 2")
    return paired


def _safe_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _training_audit(run: RunEvidence) -> dict[str, Any]:
    losses = [
        _as_float(row.get("train_loss"), f"{run.role} epoch train_loss")
        for row in run.history
    ]
    gradients = [
        _as_float(row.get("gradient_norm"), f"{run.role} epoch gradient_norm")
        for row in run.history
    ]
    all_finite = all(math.isfinite(value) for value in (*losses, *gradients))
    declared_finite = run.convergence.get("all_losses_and_gradients_finite")
    if not isinstance(declared_finite, bool) or declared_finite != all_finite:
        raise CapacityComparisonError(
            f"{run.role} convergence finiteness disagrees with epoch history"
        )
    tail_losses = losses[-20:]
    tail_gradients = gradients[-20:]
    nonfinite_tail_epochs = [
        _as_int(row.get("epoch"), f"{run.role} tail epoch")
        for row, loss, gradient in zip(
            run.history[-20:], tail_losses, tail_gradients, strict=True
        )
        if not math.isfinite(loss) or not math.isfinite(gradient)
    ]
    reasons: list[str] = []
    if nonfinite_tail_epochs:
        reasons.append(
            "nonfinite last-20 loss or gradient at epochs "
            + ", ".join(str(epoch) for epoch in nonfinite_tail_epochs)
        )

    positive_gradients = [
        value for value in gradients if math.isfinite(value) and value > 0.0
    ]
    positive_gradient_median = (
        statistics.median(positive_gradients) if positive_gradients else None
    )
    finite_tail_gradients = [
        value for value in tail_gradients if math.isfinite(value)
    ]
    tail_gradient_max = (
        max(finite_tail_gradients) if finite_tail_gradients else None
    )
    gradient_spike = bool(
        positive_gradient_median is not None
        and tail_gradient_max is not None
        and tail_gradient_max > 10.0 * positive_gradient_median
    )
    if gradient_spike:
        reasons.append(
            "last-20 gradient maximum exceeds 10x the positive all-epoch median"
        )

    same_mode_checks: list[dict[str, Any]] = []
    mode_rows: dict[str, list[tuple[int, float]]] = {}
    for row, loss in zip(run.history, losses, strict=True):
        mode_rows.setdefault(str(row.get("mask_mode")), []).append(
            (_as_int(row.get("epoch"), f"{run.role} mode epoch"), loss)
        )
    for mode in sorted(mode_rows):
        observations = mode_rows[mode]
        if len(observations) < 3:
            continue
        last_epoch, last_loss = observations[-1]
        previous = [loss for _, loss in observations[-6:-1]]
        previous_finite = [
            value for value in previous if math.isfinite(value)
        ]
        previous_median = (
            statistics.median(previous_finite)
            if len(previous_finite) == len(previous) and previous_finite
            else None
        )
        threshold = (
            1.25 * previous_median if previous_median is not None else None
        )
        divergent = bool(
            math.isfinite(last_loss)
            and threshold is not None
            and last_loss > threshold
        )
        if divergent:
            reasons.append(
                f"{mode} last loss exceeds 1.25x its previous same-mode median"
            )
        same_mode_checks.append(
            {
                "mask_mode": mode,
                "observation_count": len(observations),
                "last_epoch": last_epoch,
                "last_loss": _safe_number(last_loss),
                "previous_observations_used": len(previous),
                "previous_median_loss": (
                    _safe_number(previous_median)
                    if previous_median is not None
                    else None
                ),
                "threshold_1_25x": (
                    _safe_number(threshold) if threshold is not None else None
                ),
                "divergent": divergent,
            }
        )

    return {
        "epochs_present": len(run.history),
        "epochs_are_0_through_199": [
            _as_int(row.get("epoch"), f"{run.role} epoch")
            for row in run.history
        ]
        == list(range(_EXPECTED_EPOCHS)),
        "all_epoch_losses_and_gradients_finite": all_finite,
        "last_20_losses_and_gradients_finite": not nonfinite_tail_epochs,
        "nonfinite_last_20_epochs": nonfinite_tail_epochs,
        "positive_gradient_median_all_epochs": (
            _safe_number(positive_gradient_median)
            if positive_gradient_median is not None
            else None
        ),
        "last_20_gradient_max": (
            _safe_number(tail_gradient_max)
            if tail_gradient_max is not None
            else None
        ),
        "gradient_spike_gt_10x_positive_median": gradient_spike,
        "same_mode_loss_checks": same_mode_checks,
        "unresolved_divergence": bool(reasons),
        "divergence_reasons": reasons,
    }


def _inference_label(
    *,
    gate_passes: bool,
    relative_gain: float,
    all_replicates_favor: bool,
    finite_epochs: bool,
    no_divergence: bool,
) -> tuple[str, str]:
    if gate_passes:
        return (
            "supported_within_locked_held_in_scope",
            "The locked H1-capacity gate passes for held-in reconstruction in "
            "this one transductively fitted core.",
        )
    if relative_gain <= 0.0 and finite_epochs and no_divergence:
        return (
            "negative_nonpositive_gain",
            "H1-capacity is falsified for this configuration because the mean "
            "G2 gain is non-positive.",
        )
    if 0.0 < relative_gain < 0.02 and finite_epochs and no_divergence:
        return (
            "positive_but_subthreshold",
            "The mean G2 gain is positive but below the locked 2% capacity "
            "threshold.",
        )
    if not all_replicates_favor and finite_epochs and no_divergence:
        return (
            "inconsistent_across_technical_replicates",
            "The locked capacity gate fails because not all three paired "
            "whole-node mask replicates favor G2.",
        )
    return (
        "inconclusive_due_to_training_diagnostics",
        "The locked capacity gate fails because epoch completion, finiteness, "
        "or divergence diagnostics are unresolved.",
    )


def compare_full_core_capacity(
    g2_run: str | Path,
    matched_self_run: str | Path,
) -> dict[str, Any]:
    """Verify two canonical bundles and return the locked comparison result."""

    g2 = _load_run(g2_run, role="g2", expected_models=_G2_KEYS)
    matched_self = _load_run(
        matched_self_run,
        role="b0_g2_matched",
        expected_models=_SELF_KEYS,
    )
    parameter_count = _verify_pair(g2, matched_self)
    g2_rows = _normalized_replicates(g2)
    self_rows = _normalized_replicates(matched_self)
    _validate_replicates_against_manifest(g2, g2_rows)
    _validate_replicates_against_manifest(matched_self, self_rows)
    g2_mean = _reconcile_primary(g2, g2_rows)
    self_mean = _reconcile_primary(matched_self, self_rows)
    paired = _pair_whole_node_rows(g2_rows, self_rows)
    if self_mean <= 0.0:
        raise CapacityComparisonError(
            "matched-self mean Huber must be positive for a relative gain"
        )
    mean_difference = self_mean - g2_mean
    relative_gain = mean_difference / self_mean
    all_replicates_favor = all(row["favors_g2"] for row in paired)

    g2_training = _training_audit(g2)
    self_training = _training_audit(matched_self)
    finite_epochs = bool(
        g2_training["epochs_present"] == _EXPECTED_EPOCHS
        and self_training["epochs_present"] == _EXPECTED_EPOCHS
        and g2_training["epochs_are_0_through_199"]
        and self_training["epochs_are_0_through_199"]
        and g2_training["all_epoch_losses_and_gradients_finite"]
        and self_training["all_epoch_losses_and_gradients_finite"]
    )
    no_divergence = not bool(
        g2_training["unresolved_divergence"]
        or self_training["unresolved_divergence"]
    )
    criteria = {
        "relative_mean_gain_at_least_2_percent": relative_gain >= 0.02,
        "all_three_paired_replicates_favor_g2": all_replicates_favor,
        "both_runs_complete_200_finite_epochs": finite_epochs,
        "neither_run_has_unresolved_divergence": no_divergence,
    }
    gate_passes = all(criteria.values())
    inference_label, inference_statement = _inference_label(
        gate_passes=gate_passes,
        relative_gain=relative_gain,
        all_replicates_favor=all_replicates_favor,
        finite_epochs=finite_epochs,
        no_divergence=no_divergence,
    )
    limits = [
        "All fitted transforms, graph construction, and evaluation cells come "
        "from the same single core.",
        "The comparison is held-in and is not an estimate of generalization.",
        "The three mask replicates are technical repeats, not independent "
        "biological replicates.",
        "The qualitative no-divergence requirement was predeclared, but its "
        "exact numerical thresholds were finalized after G2 started and are "
        "therefore a post-specified audit.",
        "This analysis does not verify the earlier held-out predictive "
        "hypothesis.",
        "A capacity result does not establish cell-cell communication, a "
        "biological mechanism, or causality.",
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "full_core_capacity_comparison",
        "status": "complete",
        "inputs": {
            "g2": {
                "run_id": g2.run_id,
                "path": g2.root.as_posix(),
                "model_name": g2.model_name,
            },
            "b0_g2_matched": {
                "run_id": matched_self.run_id,
                "path": matched_self.root.as_posix(),
                "model_name": matched_self.model_name,
            },
        },
        "facts": {
            "evaluation_protocol": _PROTOCOL,
            "canonical_role": "fit",
            "model_seed": _as_int(g2.config.get("seed"), "model seed"),
            "dataset": _canonical_subset(
                _mapping(g2.config.get("dataset"), "config.dataset"),
                _PAIR_DATA_FIELDS,
            ),
            "graph_sha256": g2.summary.get("graph_sha256"),
            "evaluation_mask_bundle_sha256": g2.summary.get(
                "evaluation_mask_bundle_sha256"
            ),
            "equal_parameter_count": True,
            "parameter_count": parameter_count,
            "g2_mean_whole_node_masked_huber": g2_mean,
            "matched_self_mean_whole_node_masked_huber": self_mean,
            "mean_self_minus_g2_masked_huber": mean_difference,
            "relative_mean_g2_gain": relative_gain,
            "paired_whole_node_replicates": paired,
        },
        "training_audit": {
            "divergence_definition": {
                "nonfinite_tail": "any nonfinite loss or gradient in the last 20 epochs",
                "gradient_spike": "last-20 gradient max > 10x the positive median over all epochs",
                "same_mode_deterioration": "for a mask mode with >=3 observations, its last loss > 1.25x the median of its previous up-to-5 observations",
            },
            "g2": g2_training,
            "b0_g2_matched": self_training,
        },
        "locked_gate": {
            "thresholds": {
                "minimum_relative_mean_g2_gain": 0.02,
                "required_favorable_paired_replicates": 3,
                "required_epochs_per_run": 200,
                "allowed_unresolved_divergence": False,
            },
            "criteria": criteria,
            "passes": gate_passes,
        },
        "inference": {
            "label": inference_label,
            "statement": inference_statement,
            "maximum_claim": (
                "held-in masked-expression reconstruction capacity in one "
                "transductively fitted core"
            ),
        },
        "limits": limits,
    }


def _format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _markdown_report(result: Mapping[str, Any]) -> str:
    facts = _mapping(result["facts"], "result.facts")
    gate = _mapping(result["locked_gate"], "result.locked_gate")
    criteria = _mapping(gate["criteria"], "result.locked_gate.criteria")
    inference = _mapping(result["inference"], "result.inference")
    lines = [
        "# Full-core G2 capacity comparison",
        "",
        "## Facts",
        "",
        f"- G2 run: `{result['inputs']['g2']['run_id']}`.",
        (
            "- G2-matched cell-autonomous run: "
            f"`{result['inputs']['b0_g2_matched']['run_id']}`."
        ),
        (
            "- Both checksum-verified runs completed 200 fixed epochs with "
            f"{facts['parameter_count']:,} trainable parameters."
        ),
        (
            "- Mean held-in whole-node masked Huber: "
            f"G2 {_format_float(float(facts['g2_mean_whole_node_masked_huber']))}; "
            "matched self "
            f"{_format_float(float(facts['matched_self_mean_whole_node_masked_huber']))}."
        ),
        (
            "- Mean self-minus-G2 difference: "
            f"{_format_float(float(facts['mean_self_minus_g2_masked_huber']))}; "
            "relative G2 gain "
            f"{100.0 * float(facts['relative_mean_g2_gain']):.3f}%."
        ),
        "",
        "| Locked criterion | Pass |",
        "|---|---:|",
    ]
    labels = {
        "relative_mean_gain_at_least_2_percent": "Relative mean G2 gain ≥2%",
        "all_three_paired_replicates_favor_g2": "All three paired replicates favor G2",
        "both_runs_complete_200_finite_epochs": "Both runs complete 200 finite epochs",
        "neither_run_has_unresolved_divergence": "No unresolved divergence",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {'yes' if criteria[key] else 'no'} |")
    lines.extend(
        [
            "",
            "## Inference",
            "",
            f"**Locked gate: {'PASS' if gate['passes'] else 'FAIL'}.** "
            f"{inference['statement']}",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {limit}" for limit in result["limits"])
    lines.append("")
    return "\n".join(lines)


def write_comparison(
    result: Mapping[str, Any],
    output_directory: str | Path,
) -> Path:
    """Write comparison.json and report.md without replacing any path."""

    destination = Path(output_directory).resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise CapacityComparisonError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise CapacityComparisonError(f"output already exists: {destination}") from error
    json_path = destination / "comparison.json"
    markdown_path = destination / "report.md"
    try:
        with json_path.open("x", encoding="utf-8") as handle:
            json.dump(
                result,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        with markdown_path.open("x", encoding="utf-8") as handle:
            handle.write(_markdown_report(result))
    except BaseException as error:
        raise CapacityComparisonError(
            f"comparison output is incomplete at {destination}"
        ) from error
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and compare one canonical full-core G2 run with its "
            "G2-parameter-matched cell-autonomous control."
        )
    )
    parser.add_argument("--g2-run", required=True, type=Path)
    parser.add_argument(
        "--b0-g2-matched-run",
        "--matched-self-run",
        dest="matched_self_run",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = compare_full_core_capacity(
        arguments.g2_run,
        arguments.matched_self_run,
    )
    output = write_comparison(result, arguments.output)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "gate_passes": result["locked_gate"]["passes"],
                "inference": result["inference"]["label"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

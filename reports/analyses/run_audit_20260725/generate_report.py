#!/usr/bin/env python3
"""Build the portable registered-run audit.

The generator reads only the local registry and immutable historical artifacts.
It does not access raw or clinical inputs, mutate run artifacts, or re-evaluate
the sealed test predictions.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[2]
DATABASE = PROJECT_ROOT / "state/tracking/bagm.sqlite3"
LEGACY_ROOT = (
    PROJECT_ROOT
    / "artifacts/legacy_runs/lr_spatial_benchmark_batch/original"
)
FINAL_ANALYSIS = LEGACY_ROOT / "final_analysis_v1"
PREPARED_MANIFEST = LEGACY_ROOT / "prepared_full_v1/manifest.json"
STANDARDS_LOCK = LEGACY_ROOT / "standards_lock_v1/standards_lock.json"

STAGE_LABELS = {
    "diagnostic": "Diagnostic",
    "exploratory_screen": "Exploratory screen",
    "validation_confirmation": "Validation confirmation",
    "locked_final": "Locked final",
}
MASK_LABELS = {
    "partial": "Partial-gene",
    "node": "Whole-node",
    "block": "Spatial block",
}
CONDITION_ORDER = [
    "b0",
    "b0_parameter_matched",
    "broad_field",
    "b1",
    "g1_true",
    "g1_rewired",
    "g2_true",
    "g2_zero",
    "g2_distance_only",
    "g2_permuted",
]
MODEL_ORDER = ["b0", "b0-matched", "broad-field", "b1", "g1", "g2"]


class Raw(str):
    """HTML fragment that has already been escaped or deliberately authored."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolved_artifact_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def h(value: Any) -> str:
    if isinstance(value, Raw):
        return str(value)
    if value is None:
        return "—"
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "—"
    number = float(value)
    return f"{number:.{digits}f}"


def pct(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}%"


def human_bytes(value: int | float) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024.0
    raise AssertionError("unreachable")


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    caption: str | None = None,
    classes: str = "",
    table_id: str | None = None,
) -> str:
    identifier = f' id="{h(table_id)}"' if table_id else ""
    chunks = [f'<div class="table-wrap"><table class="{h(classes)}"{identifier}>']
    if caption:
        chunks.append(f"<caption>{h(caption)}</caption>")
    chunks.append("<thead><tr>")
    for heading in headers:
        chunks.append(f'<th scope="col">{h(heading)}</th>')
    chunks.append("</tr></thead><tbody>")
    for row in rows:
        chunks.append("<tr>")
        for value in row:
            chunks.append(f"<td>{h(value)}</td>")
        chunks.append("</tr>")
    chunks.append("</tbody></table></div>")
    return "".join(chunks)


def badge(label: str, kind: str) -> Raw:
    return Raw(f'<span class="badge {h(kind)}">{h(label)}</span>')


def model_condition(
    model_name: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    explicit = (
        manifest.get("standards_lock", {}).get("condition")
        if isinstance(manifest.get("standards_lock"), Mapping)
        else None
    )
    if isinstance(explicit, str) and explicit:
        return explicit.lower().replace("-", "_")
    graph = config.get("graph", {})
    native_graph = manifest.get("graph", {})
    edge_control = str(
        native_graph.get("edge_control", graph.get("edge_control", "none"))
    )
    rewired = bool(
        graph.get("rewired")
        or native_graph.get("kind") == "rewired"
        or (
            isinstance(native_graph.get("rewire"), Mapping)
            and native_graph["rewire"].get("enabled")
        )
    )
    if model_name == "b0-matched":
        return "b0_parameter_matched"
    if model_name == "broad-field":
        return "broad_field"
    if rewired:
        return f"{model_name}_rewired"
    if edge_control != "none":
        return f"{model_name}_{edge_control}"
    if model_name in {"g1", "g2", "g3"}:
        return f"{model_name}_true"
    return model_name


def condition_role(condition: str) -> str:
    if condition in {"b0", "b0_parameter_matched", "b1", "broad_field"}:
        return "baseline"
    if condition.endswith("_rewired"):
        return "mechanism-breaking control"
    if condition.endswith(("_zero", "_distance_only", "_permuted")):
        return "edge-feature control"
    return "candidate model"


def checkpoint_parameter_count(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")
    # Broad-Field stores nine train-fitted coordinate-basis values as buffers.
    # Other state tensors in these model families are trainable parameters.
    return sum(
        int(tensor.numel())
        for name, tensor in state.items()
        if torch.is_tensor(tensor) and not str(name).startswith("_coordinate_")
    )


def load_run_rows() -> list[dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{DATABASE}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    query = """
        SELECT
            r.run_id, r.campaign_id, r.scientific_id, r.repro_id,
            r.status, r.model_family, r.masking_type, r.dataset_id,
            r.dataset_version, r.split_id, r.seed, r.fold, r.attempt,
            r.git_commit, r.dirty_status, r.start_time, r.end_time,
            r.duration_seconds, r.gpu_model, r.peak_vram_gb,
            r.primary_metric_name, r.primary_metric_value,
            r.artifact_path, r.config_json,
            c.lifecycle_stage, c.study_axis, c.source_batch,
            c.variant_label, c.model_key, c.dataset_key, c.masking_key,
            c.graph_key, c.feature_key, c.embedding_key,
            c.seed_known, c.fold_known, c.attempt_known,
            c.retention_class, c.classification_confidence,
            a.alias_id,
            cc.role AS checkpoint_role, cc.best_epoch,
            cc.monitored_metric, cc.monitored_mode, cc.monitored_value,
            cc.duplicate_group, cc.duplicate_count,
            cc.verification_status, cc.metadata_json AS checkpoint_metadata,
            ck.path AS checkpoint_path, ck.sha256 AS checkpoint_sha256,
            ck.size_bytes AS checkpoint_size_bytes
        FROM runs r
        JOIN run_categories c USING (run_id)
        JOIN run_aliases a
          ON a.run_id = r.run_id AND a.preferred = 1
        JOIN checkpoint_catalog cc USING (run_id)
        JOIN artifacts ck ON ck.artifact_id = cc.artifact_id
        ORDER BY
            CASE c.lifecycle_stage
                WHEN 'diagnostic' THEN 0
                WHEN 'exploratory_screen' THEN 1
                WHEN 'validation_confirmation' THEN 2
                WHEN 'locked_final' THEN 3
                ELSE 4
            END,
            c.study_axis, r.model_family, r.seed, r.run_id
    """
    source_rows = connection.execute(query).fetchall()
    artifact_rows = connection.execute(
        """
        SELECT run_id, kind, path, sha256, size_bytes, status
        FROM artifacts ORDER BY run_id, kind
        """
    ).fetchall()
    connection.close()

    artifacts: dict[str, dict[str, sqlite3.Row]] = defaultdict(dict)
    for artifact in artifact_rows:
        artifacts[str(artifact["run_id"])][str(artifact["kind"])] = artifact

    output: list[dict[str, Any]] = []
    for source in source_rows:
        row = dict(source)
        run_artifacts = artifacts[row["run_id"]]
        manifest_record = run_artifacts["legacy_manifest"]
        metrics_record = run_artifacts["legacy_metrics"]
        manifest_path = resolved_artifact_path(str(manifest_record["path"]))
        metrics_path = resolved_artifact_path(str(metrics_record["path"]))
        checkpoint_path = resolved_artifact_path(str(row["checkpoint_path"]))
        manifest = read_json(manifest_path)
        config = json.loads(row["config_json"])
        checkpoint_metadata = json.loads(row["checkpoint_metadata"])
        trainer = config.get("trainer", {})
        graph = config.get("graph", {})
        condition = model_condition(row["model_family"], config, manifest)
        edge_control = str(
            manifest.get("graph", {}).get(
                "edge_control", graph.get("edge_control", "none")
            )
        )
        rewired = bool(
            graph.get("rewired")
            or manifest.get("graph", {}).get("kind") == "rewired"
        )
        parameter_count = checkpoint_parameter_count(checkpoint_path)
        output.append(
            {
                **row,
                "condition": condition,
                "condition_role": condition_role(condition),
                "dataset_version": config.get("dataset", {}).get(
                    "version", row["dataset_version"]
                ),
                "masking": config.get("masking", {}).get(
                    "type", row["masking_type"]
                ),
                "graph_id": graph.get("graph_id", row["graph_key"]),
                "graph_k": graph.get("neighbor_k"),
                "graph_radius_um": graph.get("radius_um"),
                "graph_symmetry": graph.get("symmetry"),
                "rewired": rewired,
                "edge_control": edge_control,
                "hidden_dim": trainer.get("embedding_dim"),
                "graph_layers": trainer.get("graph_layers"),
                "edge_embedding_dim": trainer.get("edge_embedding_dim"),
                "attention_heads": trainer.get("attention_heads"),
                "learning_rate": trainer.get("learning_rate"),
                "weight_decay": trainer.get("weight_decay"),
                "max_epochs": trainer.get("max_epochs"),
                "parameter_count": parameter_count,
                "checkpoint_path_relative": relative_path(checkpoint_path),
                "manifest_path_relative": relative_path(manifest_path),
                "metrics_path_relative": relative_path(metrics_path),
                "artifact_path_relative": relative_path(
                    resolved_artifact_path(str(row["artifact_path"]))
                ),
                "predictions_present": "legacy_predictions_restricted"
                in run_artifacts,
                "checkpoint_id": checkpoint_metadata.get("checkpoint_id"),
                "evidence_tier": checkpoint_metadata.get("evidence_tier"),
                "interpretation_tier": checkpoint_metadata.get(
                    "interpretation_tier"
                ),
                "fold_display": (
                    str(row["fold"]) if row["fold_known"] else "unknown"
                ),
                "attempt_display": (
                    str(row["attempt"]) if row["attempt_known"] else "unknown"
                ),
            }
        )
    return output


CSV_FIELDS = [
    "run_id",
    "alias_id",
    "lifecycle_stage",
    "study_axis",
    "status",
    "model_family",
    "condition",
    "condition_role",
    "seed",
    "fold_display",
    "attempt_display",
    "dataset_id",
    "dataset_version",
    "split_id",
    "masking",
    "graph_id",
    "graph_k",
    "graph_radius_um",
    "graph_symmetry",
    "rewired",
    "edge_control",
    "hidden_dim",
    "graph_layers",
    "edge_embedding_dim",
    "attention_heads",
    "learning_rate",
    "weight_decay",
    "max_epochs",
    "parameter_count",
    "checkpoint_size_bytes",
    "best_epoch",
    "primary_metric_name",
    "primary_metric_value",
    "duration_seconds",
    "peak_vram_gb",
    "gpu_model",
    "predictions_present",
    "verification_status",
    "duplicate_group",
    "duplicate_count",
    "evidence_tier",
    "interpretation_tier",
    "git_commit",
    "dirty_status",
    "checkpoint_sha256",
    "checkpoint_path_relative",
    "manifest_path_relative",
    "metrics_path_relative",
    "artifact_path_relative",
]


def write_runs_csv(rows: Sequence[Mapping[str, Any]]) -> Path:
    path = REPORT_DIR / "runs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    return path


def artifact_summary() -> dict[str, Any]:
    connection = sqlite3.connect(
        f"file:{DATABASE}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    groups = [
        dict(row)
        for row in connection.execute(
            """
            SELECT kind, COUNT(*) AS count,
                   SUM(COALESCE(size_bytes, 0)) AS size_bytes
            FROM artifacts GROUP BY kind ORDER BY kind
            """
        )
    ]
    total = dict(
        connection.execute(
            """
            SELECT COUNT(*) AS count,
                   SUM(COALESCE(size_bytes, 0)) AS size_bytes
            FROM artifacts
            """
        ).fetchone()
    )
    connection.close()
    return {"groups": groups, "total": total}


def final_condition_resources(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["lifecycle_stage"] == "locked_final":
            grouped[str(row["condition"])].append(row)
    output = []
    for condition in CONDITION_ORDER:
        values = grouped[condition]
        output.append(
            {
                "condition": condition,
                "n": len(values),
                "parameter_count": sorted(
                    {int(value["parameter_count"]) for value in values}
                ),
                "checkpoint_size_bytes": sorted(
                    {int(value["checkpoint_size_bytes"]) for value in values}
                ),
                "best_epoch_min": min(int(value["best_epoch"]) for value in values),
                "best_epoch_median": statistics.median(
                    int(value["best_epoch"]) for value in values
                ),
                "best_epoch_max": max(int(value["best_epoch"]) for value in values),
                "duration_mean": statistics.mean(
                    float(value["duration_seconds"]) for value in values
                ),
                "peak_vram_max": max(
                    float(value["peak_vram_gb"]) for value in values
                ),
            }
        )
    return output


def screen_decisions() -> list[dict[str, Any]]:
    specifications = [
        ("Mask curriculum", "selection_mask_v1/recommendation.json"),
        ("Graph", "selection_graph_v1/recommendation.json"),
        ("Hidden width", "selection_hidden_v1/recommendation.json"),
        ("G1 depth", "selection_depth_v1/recommendation.json"),
        (
            "G2 edge embedding",
            "selection_edge_embedding_v1/recommendation.json",
        ),
        (
            "Post-lock confirmation",
            "summary_postlock_confirmation_v1/recommendation.json",
        ),
    ]
    output = []
    for label, relative in specifications:
        record = read_json(LEGACY_ROOT / relative)
        standard = ", ".join(
            f"{key}={value}"
            for key, value in record["standard"].items()
            if value is not None
        )
        top = record["ranked_candidates"][0]
        runner_up = (
            record["ranked_candidates"][1]
            if len(record["ranked_candidates"]) > 1
            else None
        )
        output.append(
            {
                "stage": label,
                "standard": standard,
                "whole_node_huber": float(record["whole_node_huber"]),
                "block_huber": float(record["spatial_block_huber"]),
                "seed_sd": float(record["whole_node_seed_sd"]),
                "required_seeds": int(record["required_seeds"]),
                "runner_up_whole_node": (
                    float(runner_up["whole_node_huber"])
                    if runner_up is not None
                    else None
                ),
                "test_metrics_used": bool(record["test_metrics_used"]),
                "candidate_id": top["candidate_id"],
            }
        )
    return output


def make_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage_counts = Counter(str(row["lifecycle_stage"]) for row in rows)
    axis_counts = Counter(str(row["study_axis"]) for row in rows)
    model_counts = Counter(str(row["model_family"]) for row in rows)
    final_losses = [
        row
        for row in read_csv(FINAL_ANALYSIS / "model_mode_losses.csv")
        if row["split"] == "test"
    ]
    gains = [
        row
        for row in read_csv(FINAL_ANALYSIS / "spatial_gains.csv")
        if row["split"] == "test"
    ]
    gate = read_csv(FINAL_ANALYSIS / "acceptance_gate.csv")
    artifacts = artifact_summary()
    duplicate_groups = {
        str(row["duplicate_group"])
        for row in rows
        if row["duplicate_group"]
    }
    prepared = read_json(PREPARED_MANIFEST)
    lock = read_json(STANDARDS_LOCK)
    primary_gain = next(
        row
        for row in gains
        if row["comparison"] == "B0_minus_G1"
        and row["mask_mode"] == "node"
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "All runs in the local BAGM SQLite registry",
        "run_count": len(rows),
        "stage_counts": dict(sorted(stage_counts.items())),
        "study_axis_counts": dict(sorted(axis_counts.items())),
        "model_counts": dict(sorted(model_counts.items())),
        "status_counts": dict(
            sorted(Counter(str(row["status"]) for row in rows).items())
        ),
        "dataset_version_counts": dict(
            sorted(
                Counter(str(row["dataset_version"]) for row in rows).items()
            )
        ),
        "prediction_bearing_runs": sum(
            bool(row["predictions_present"]) for row in rows
        ),
        "checkpoint_verification_counts": dict(
            sorted(
                Counter(
                    str(row["verification_status"]) for row in rows
                ).items()
            )
        ),
        "duplicate_checkpoint_groups": len(duplicate_groups),
        "duplicate_checkpoint_records": sum(
            bool(row["duplicate_group"]) for row in rows
        ),
        "artifact_summary": artifacts,
        "aggregate_recorded_run_seconds": sum(
            float(row["duration_seconds"]) for row in rows
        ),
        "max_peak_vram_gb": max(float(row["peak_vram_gb"]) for row in rows),
        "parameter_count_range": [
            min(int(row["parameter_count"]) for row in rows),
            max(int(row["parameter_count"]) for row in rows),
        ],
        "checkpoint_size_range": [
            min(int(row["checkpoint_size_bytes"]) for row in rows),
            max(int(row["checkpoint_size_bytes"]) for row in rows),
        ],
        "final_condition_resources": final_condition_resources(rows),
        "screen_decisions": screen_decisions(),
        "primary_result": {
            "baseline_loss": float(primary_gain["baseline_loss"]),
            "g1_loss": float(primary_gain["spatial_loss"]),
            "delta": float(primary_gain["delta"]),
            "relative_gain_percent": float(
                primary_gain["relative_gain_percent"]
            ),
            "ci_lower": float(primary_gain["delta_ci_lower"]),
            "ci_upper": float(primary_gain["delta_ci_upper"]),
            "prespecified_threshold_percent": 2.0,
            "gate_supported": False,
        },
        "final_test_losses": final_losses,
        "paired_test_gains": gains,
        "acceptance_gate": gate,
        "prepared_data": {
            "artifact_id": prepared["artifact_id"],
            "cells": prepared["selection"]["n_cells"],
            "fovs": prepared["selection"]["n_fovs"],
            "genes": prepared["features"]["n_biological_probes"],
            "covariates": len(prepared["features"]["model_covariate_names"]),
            "split_counts": prepared["split"]["counts"],
            "macroblocks": prepared["split"]["n_macroblocks"],
            "split_id": prepared["split"]["split_id"],
            "validation_mask_replicates": prepared["fixed_masks"]["bundles"][
                "validation"
            ]["replicates"],
            "test_mask_replicates": prepared["fixed_masks"]["bundles"]["test"][
                "replicates"
            ],
            "model_covariate_names": prepared["features"][
                "model_covariate_names"
            ],
            "technical_control_prefixes_excluded": prepared["features"][
                "technical_control_prefixes_excluded"
            ],
        },
        "standards": lock["standards"],
        "provenance": {
            "database": relative_path(DATABASE),
            "database_sha256": sha256_file(DATABASE),
            "prepared_manifest": relative_path(PREPARED_MANIFEST),
            "prepared_manifest_sha256": sha256_file(PREPARED_MANIFEST),
            "standards_lock": relative_path(STANDARDS_LOCK),
            "standards_lock_sha256": sha256_file(STANDARDS_LOCK),
            "final_summary": relative_path(FINAL_ANALYSIS / "summary.json"),
            "final_summary_sha256": sha256_file(
                FINAL_ANALYSIS / "summary.json"
            ),
            "git_commit": rows[0]["git_commit"],
            "historical_worktree_dirty": bool(rows[0]["dirty_status"]),
        },
    }


def write_summary(summary: Mapping[str, Any]) -> Path:
    path = REPORT_DIR / "summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def bar_rows(counts: Mapping[str, int], labels: Mapping[str, str]) -> str:
    maximum = max(counts.values())
    chunks = ['<div class="bars" role="img" aria-label="Run counts">']
    for key, count in counts.items():
        label = labels.get(key, key.replace("_", " ").title())
        width = 100.0 * count / maximum
        chunks.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{h(label)}</span>'
            f'<span class="bar-track"><span class="bar-fill" '
            f'style="width:{width:.2f}%"></span></span>'
            f'<strong>{count}</strong></div>'
        )
    chunks.append("</div>")
    return "".join(chunks)


def render_model_catalog(
    summary: Mapping[str, Any],
) -> str:
    resources = {
        row["condition"]: row for row in summary["final_condition_resources"]
    }
    descriptions = [
        (
            "B0",
            "b0",
            "Self-only MLP",
            "Masked expression + explicit gene mask + 22 measured morphology/imaging covariates. No coordinates or neighbors.",
            "Cell-autonomous baseline.",
        ),
        (
            "B0 matched",
            "b0_parameter_matched",
            "Parameter-matched self-only network",
            "Two self-only mixing blocks shaped to have exactly the same trainable parameter count as locked G1.",
            "Tests whether G1 gains are merely capacity.",
        ),
        (
            "Broad-Field",
            "broad_field",
            "B0 plus global quadratic field",
            "Adds train-fitted x, y, x², xy, y² features; still has no graph or neighbor aggregation.",
            "Tests broad location/FOV-scale structure.",
        ),
        (
            "B1",
            "b1",
            "Uniform mean-neighbor model",
            "Projects the arithmetic mean of incoming neighbor embeddings, then applies residual feed-forward decoding.",
            "Tests simple spatial smoothing.",
        ),
        (
            "G1",
            "g1_true",
            "Topology-only GATv2",
            "Two graph layers, four attention heads, learned neighbor routing, no edge attributes, and no self-loops.",
            "Primary local-graph candidate.",
        ),
        (
            "G2",
            "g2_true",
            "Edge-conditioned GATv2",
            "G1 plus a shared encoder mapping 17 geometric edge features into a 64-dimensional edge embedding used by attention.",
            "Tests added value from measured geometry.",
        ),
        (
            "G3",
            None,
            "Additive self + signed edge messages",
            "Planned interpretable model with an exact self/neighbor decomposition.",
            "Not trained: its validation utility gate was not met.",
        ),
    ]
    rows = []
    for label, key, architecture, inputs, purpose in descriptions:
        if key is None:
            params = "Not trained"
            size = "—"
            count = 0
        else:
            record = resources[key]
            params = f"{record['parameter_count'][0]:,}"
            size = human_bytes(record["checkpoint_size_bytes"][0])
            family_key = {
                "b0_parameter_matched": "b0-matched",
                "broad_field": "broad-field",
            }.get(key, key.split("_")[0])
            count = summary["model_counts"].get(family_key, 0)
        rows.append(
            [label, architecture, inputs, purpose, params, size, count]
        )
    return table(
        [
            "Model",
            "Architecture",
            "Inputs / operation",
            "Experimental role",
            "Locked parameters",
            "Checkpoint",
            "All runs",
        ],
        rows,
        caption=(
            "Model-family basics. Sizes are reconstructed from checkpoint "
            "state dictionaries; checkpoint size is serialized on disk."
        ),
        classes="model-table",
    )


def render_selection_table(summary: Mapping[str, Any]) -> str:
    rows = []
    for item in summary["screen_decisions"]:
        runner_up = (
            "—"
            if item["runner_up_whole_node"] is None
            else fmt(item["runner_up_whole_node"])
        )
        rows.append(
            [
                item["stage"],
                item["standard"],
                item["required_seeds"],
                fmt(item["whole_node_huber"]),
                runner_up,
                fmt(item["block_huber"]),
                fmt(item["seed_sd"]),
                "No" if not item["test_metrics_used"] else "Yes",
            ]
        )
    return table(
        [
            "Screen",
            "Selected standard",
            "Seeds",
            "Whole-node Huber",
            "Runner-up",
            "Block Huber",
            "Seed SD",
            "Test used?",
        ],
        rows,
        caption=(
            "Sequential validation-only decisions. These stages changed more "
            "than one nuisance context over time and are not a Cartesian "
            "cross-stage comparison."
        ),
    )


def render_final_loss_table(summary: Mapping[str, Any]) -> str:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    labels: dict[str, str] = {}
    intervals: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in summary["final_test_losses"]:
        condition = row["condition"]
        mode = row["mask_mode"]
        values[condition][mode] = float(row["huber_loss"])
        intervals[condition][mode] = (
            float(row["huber_ci_lower"]),
            float(row["huber_ci_upper"]),
        )
        labels[condition] = row["condition_label"]
    minima = {
        mode: min(values[condition][mode] for condition in CONDITION_ORDER)
        for mode in ("partial", "node", "block")
    }
    rows = []
    for condition in CONDITION_ORDER:
        cells: list[Any] = [labels[condition]]
        for mode in ("partial", "node", "block"):
            value = values[condition][mode]
            rendered = fmt(value)
            if value == minima[mode]:
                rendered = Raw(
                    f'<strong>{h(rendered)}</strong>'
                    '<span class="sr-only">, lowest point estimate</span>'
                )
            cells.append(rendered)
        rows.append(cells)
    return table(
        ["Condition", "Partial-gene", "Whole-node", "Spatial block"],
        rows,
        caption=(
            "Locked test Huber loss after five-seed ensembling, fixed-mask "
            "averaging, and equal weighting of spatial blocks. Lower is "
            "better. Bold marks the lowest point estimate, not a statistically "
            "established global winner."
        ),
        classes="numeric",
    )


def render_pairwise_table(summary: Mapping[str, Any]) -> str:
    rows = []
    for item in summary["paired_test_gains"]:
        if item["mask_mode"] != "node":
            continue
        delta = float(item["delta"])
        lower = float(item["delta_ci_lower"])
        upper = float(item["delta_ci_upper"])
        supported = lower > 0
        rows.append(
            [
                item["baseline_condition"].replace("_", " "),
                item["spatial_condition"].replace("_", " "),
                fmt(delta),
                f"[{fmt(lower)}, {fmt(upper)}]",
                pct(item["relative_gain_percent"]),
                badge(
                    "interval > 0" if supported else "interval crosses 0",
                    "pass" if supported else "neutral",
                ),
            ]
        )
    return table(
        [
            "Reference",
            "Compared model",
            "Reference − compared",
            "95% spatial-block interval",
            "Relative gain",
            "Descriptive result",
        ],
        rows,
        caption=(
            "Whole-node paired contrasts. Positive values favor the compared "
            "model. Intervals resample spatial blocks; they do not establish "
            "patient-level replication."
        ),
    )


def render_gate_table(summary: Mapping[str, Any]) -> str:
    rows = []
    for item in summary["acceptance_gate"]:
        if item["category"] != "hypothesis_gate":
            continue
        passed = item["passed"].lower() == "true"
        rows.append(
            [
                item["criterion"].replace("_", " "),
                item["observed"],
                item["threshold"],
                badge("Pass" if passed else "Fail", "pass" if passed else "fail"),
            ]
        )
    return table(
        ["Prespecified criterion", "Observed", "Requirement", "Result"],
        rows,
        caption="All four scientific criteria were required; one failed.",
    )


def render_resource_table(summary: Mapping[str, Any]) -> str:
    label_map = {
        row["condition"]: next(
            (
                loss["condition_label"]
                for loss in summary["final_test_losses"]
                if loss["condition"] == row["condition"]
            ),
            row["condition"],
        )
        for row in summary["final_condition_resources"]
    }
    rows = []
    for item in summary["final_condition_resources"]:
        rows.append(
            [
                label_map[item["condition"]],
                item["n"],
                f"{item['parameter_count'][0]:,}",
                human_bytes(item["checkpoint_size_bytes"][0]),
                (
                    f"{item['best_epoch_min']}–{item['best_epoch_max']} "
                    f"(median {item['best_epoch_median']:g})"
                ),
                f"{item['duration_mean']:.1f} s",
                f"{item['peak_vram_max']:.2f} GiB",
            ]
        )
    return table(
        [
            "Locked condition",
            "Runs",
            "Parameters",
            "Checkpoint",
            "Best epoch (0-based)",
            "Mean run time",
            "Max peak VRAM",
        ],
        rows,
        caption=(
            "Recorded resources for the 50 locked runs. G1 rewiring time "
            "includes its control-graph preparation and is not a pure model "
            "throughput benchmark."
        ),
    )


def render_run_table(rows: Sequence[Mapping[str, Any]]) -> str:
    chunks = [
        '<div class="filters" role="search">',
        '<label>Search <input id="runSearch" type="search" '
        'placeholder="alias, run ID, graph, condition…"></label>',
        '<label>Stage <select id="stageFilter"><option value="">All</option>',
    ]
    for stage in STAGE_LABELS:
        chunks.append(
            f'<option value="{h(stage)}">{h(STAGE_LABELS[stage])}</option>'
        )
    chunks.extend(
        [
            "</select></label>",
            '<label>Model <select id="modelFilter"><option value="">All</option>',
        ]
    )
    for model in MODEL_ORDER:
        chunks.append(f'<option value="{h(model)}">{h(model)}</option>')
    chunks.extend(
        [
            "</select></label>",
            '<label>Axis <select id="axisFilter"><option value="">All</option>',
        ]
    )
    for axis in sorted({str(row["study_axis"]) for row in rows}):
        chunks.append(
            f'<option value="{h(axis)}">{h(axis.replace("_", " "))}</option>'
        )
    chunks.extend(
        [
            "</select></label>",
            f'<span id="visibleRuns" aria-live="polite">{len(rows)} runs</span>',
            "</div>",
            '<div class="table-wrap run-table-wrap"><table id="runTable" '
            'class="run-table"><caption>Complete registered-run inventory. '
            "The checkpoint metric is the per-run validation selection metric, "
            "not the locked seed-ensemble test result.</caption>",
            "<thead><tr>",
        ]
    )
    headers = [
        "Stage / axis",
        "Model / condition",
        "Seed",
        "Configuration",
        "Parameters",
        "Best epoch",
        "Validation Huber",
        "Runtime / VRAM",
        "Checkpoint",
        "Run identity",
    ]
    for heading in headers:
        chunks.append(f'<th scope="col">{h(heading)}</th>')
    chunks.append("</tr></thead><tbody>")
    for row in rows:
        config_bits = [
            f"data {row['dataset_version']}",
            f"mask {row['masking']}",
            f"graph {row['graph_id']}",
            f"d={row['hidden_dim']}",
        ]
        if row["graph_layers"] is not None:
            config_bits.append(f"layers={row['graph_layers']}")
        if row["edge_embedding_dim"] is not None:
            config_bits.append(f"edge-d={row['edge_embedding_dim']}")
        if row["rewired"]:
            config_bits.append("rewired")
        if row["edge_control"] != "none":
            config_bits.append(f"control={row['edge_control']}")
        stage_badge = badge(
            STAGE_LABELS.get(row["lifecycle_stage"], row["lifecycle_stage"]),
            row["lifecycle_stage"],
        )
        duplicate = (
            '<br><span class="muted">duplicate-content group</span>'
            if row["duplicate_group"]
            else ""
        )
        search_text = " ".join(
            str(row.get(key, ""))
            for key in (
                "run_id",
                "alias_id",
                "study_axis",
                "model_family",
                "condition",
                "graph_id",
                "dataset_version",
            )
        ).lower()
        chunks.append(
            f'<tr data-run="1" data-stage="{h(row["lifecycle_stage"])}" '
            f'data-model="{h(row["model_family"])}" '
            f'data-axis="{h(row["study_axis"])}" '
            f'data-search="{h(search_text)}">'
            f"<td>{stage_badge}<br>{h(str(row['study_axis']).replace('_', ' '))}</td>"
            f"<td><strong>{h(row['model_family'])}</strong><br>"
            f"{h(str(row['condition']).replace('_', ' '))}</td>"
            f"<td>{h(row['seed'])}<br><span class=\"muted\">fold "
            f"{h(row['fold_display'])}; attempt {h(row['attempt_display'])}</span></td>"
            f"<td>{h('; '.join(config_bits))}</td>"
            f"<td>{int(row['parameter_count']):,}</td>"
            f"<td>{h(row['best_epoch'])}</td>"
            f"<td>{fmt(row['primary_metric_value'])}</td>"
            f"<td>{float(row['duration_seconds']):.1f} s<br>"
            f"{float(row['peak_vram_gb']):.2f} GiB</td>"
            f"<td>{h(human_bytes(row['checkpoint_size_bytes']))}<br>"
            f"{badge('verified', 'pass')}{duplicate}</td>"
            f"<td><code>{h(row['alias_id'])}</code><br>"
            f"<span class=\"muted\"><code>{h(row['run_id'])}</code></span></td>"
            "</tr>"
        )
    chunks.append("</tbody></table></div>")
    return "".join(chunks)


CSS = r"""
:root {
  --ink: #16202a;
  --muted: #596675;
  --line: #d9e0e7;
  --paper: #ffffff;
  --soft: #f4f7fa;
  --blue: #155eef;
  --blue-soft: #eaf1ff;
  --green: #087a55;
  --green-soft: #e6f6ef;
  --amber: #8a5700;
  --amber-soft: #fff3d6;
  --red: #b42318;
  --red-soft: #feeceb;
  --purple: #6938a5;
  --purple-soft: #f3edfb;
  --max: 1180px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: #edf2f6;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  line-height: 1.55;
}
header, main, footer {
  width: min(var(--max), calc(100% - 32px));
  margin-inline: auto;
}
header { padding: 56px 0 28px; }
main { padding-bottom: 48px; }
footer {
  color: var(--muted);
  padding: 0 0 48px;
  font-size: .9rem;
}
h1, h2, h3 { line-height: 1.16; letter-spacing: -.02em; }
h1 { font-size: clamp(2rem, 5vw, 3.65rem); margin: 0 0 14px; max-width: 900px; }
h2 { font-size: clamp(1.45rem, 3vw, 2rem); margin: 0 0 16px; }
h3 { font-size: 1.12rem; margin: 0 0 8px; }
p { margin: 0 0 14px; }
a { color: var(--blue); }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .88em;
  overflow-wrap: anywhere;
}
.eyebrow {
  color: var(--blue);
  font-weight: 750;
  text-transform: uppercase;
  letter-spacing: .09em;
  font-size: .78rem;
}
.lede { font-size: 1.14rem; color: var(--muted); max-width: 850px; }
.meta { color: var(--muted); font-size: .9rem; }
nav {
  display: flex; gap: 8px; flex-wrap: wrap; margin-top: 24px;
}
nav a {
  text-decoration: none; color: var(--ink); background: var(--paper);
  border: 1px solid var(--line); border-radius: 999px; padding: 7px 12px;
  font-size: .88rem;
}
section {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: clamp(20px, 4vw, 38px);
  margin: 18px 0;
  box-shadow: 0 8px 28px rgba(25, 39, 52, .045);
}
.verdict {
  border-left: 7px solid var(--red);
  background: linear-gradient(120deg, var(--red-soft), #fff 62%);
}
.verdict h2 { color: var(--red); }
.grid { display: grid; gap: 16px; }
.grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.stat {
  border: 1px solid var(--line); background: var(--soft);
  border-radius: 12px; padding: 16px;
}
.stat strong { display: block; font-size: 1.65rem; line-height: 1.1; }
.stat span { color: var(--muted); font-size: .88rem; }
.callout {
  border: 1px solid #b9cdfd; border-left: 5px solid var(--blue);
  background: var(--blue-soft); border-radius: 10px; padding: 15px 17px;
  margin: 16px 0;
}
.callout.warning {
  border-color: #f0cf87; border-left-color: var(--amber);
  background: var(--amber-soft);
}
.callout.negative {
  border-color: #f3b4ae; border-left-color: var(--red);
  background: var(--red-soft);
}
.facts {
  display: grid; grid-template-columns: 110px 1fr; gap: 8px 16px;
  margin: 16px 0;
}
.facts dt { font-weight: 750; }
.facts dd { margin: 0; color: var(--muted); }
.bars { display: grid; gap: 10px; margin: 18px 0; }
.bar-row {
  display: grid; grid-template-columns: 180px 1fr 44px;
  gap: 12px; align-items: center;
}
.bar-label { font-size: .9rem; }
.bar-track { height: 12px; border-radius: 99px; background: #e7edf3; overflow: hidden; }
.bar-fill { display: block; height: 100%; border-radius: inherit; background: var(--blue); }
.table-wrap { overflow-x: auto; margin: 18px 0; }
table {
  width: 100%; border-collapse: collapse; font-size: .88rem;
  font-variant-numeric: tabular-nums;
}
caption { text-align: left; color: var(--muted); padding: 0 0 10px; }
th, td {
  border-bottom: 1px solid var(--line); padding: 10px 11px;
  text-align: left; vertical-align: top;
}
thead th {
  position: sticky; top: 0; background: #f7f9fb; z-index: 1;
  white-space: nowrap;
}
tbody tr:hover { background: #f8fbff; }
.numeric td:not(:first-child) { text-align: right; }
.badge {
  display: inline-block; border-radius: 999px; padding: 3px 8px;
  font-size: .72rem; line-height: 1.2; font-weight: 750; white-space: nowrap;
}
.badge.pass, .badge.locked_final { color: var(--green); background: var(--green-soft); }
.badge.fail { color: var(--red); background: var(--red-soft); }
.badge.neutral, .badge.exploratory_screen { color: var(--amber); background: var(--amber-soft); }
.badge.diagnostic { color: var(--muted); background: #e8edf2; }
.badge.validation_confirmation { color: var(--purple); background: var(--purple-soft); }
figure { margin: 24px 0; }
figure img {
  display: block; width: 100%; height: auto;
  border: 1px solid var(--line); border-radius: 10px; background: white;
}
figcaption { color: var(--muted); font-size: .86rem; margin-top: 8px; }
.pipeline { counter-reset: step; list-style: none; padding: 0; margin: 18px 0; }
.pipeline li {
  counter-increment: step; position: relative; padding: 0 0 18px 48px;
}
.pipeline li::before {
  content: counter(step); position: absolute; left: 0; top: -2px;
  width: 30px; height: 30px; display: grid; place-items: center;
  border-radius: 50%; background: var(--blue); color: white; font-weight: 800;
}
.pipeline li:not(:last-child)::after {
  content: ""; position: absolute; left: 14px; top: 31px;
  bottom: 2px; border-left: 2px solid #b9cdfd;
}
.hypothesis {
  border-left: 4px solid var(--line); padding-left: 14px; margin: 16px 0;
}
.hypothesis.support { border-color: var(--green); }
.hypothesis.negative { border-color: var(--red); }
.hypothesis.uncertain { border-color: var(--amber); }
.filters {
  display: flex; flex-wrap: wrap; gap: 12px; align-items: end;
  margin: 18px 0;
}
.filters label { display: grid; gap: 4px; font-size: .78rem; font-weight: 700; }
.filters input, .filters select {
  min-height: 38px; border: 1px solid #aeb9c5; border-radius: 7px;
  background: white; color: var(--ink); padding: 6px 9px;
}
.filters input { min-width: min(360px, 70vw); }
#visibleRuns { color: var(--muted); padding: 8px 0; }
.run-table { min-width: 1750px; font-size: .78rem; }
.run-table-wrap { max-height: 720px; border: 1px solid var(--line); border-radius: 8px; }
.muted { color: var(--muted); }
.small { font-size: .84rem; }
.sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}
details > summary { cursor: pointer; font-weight: 750; }
ul.tight li { margin: 5px 0; }
@media (max-width: 820px) {
  .grid.two, .grid.three, .grid.four { grid-template-columns: 1fr; }
  .bar-row { grid-template-columns: 120px 1fr 36px; }
  .facts { grid-template-columns: 1fr; }
  .facts dd { margin-bottom: 8px; }
}
@media print {
  body { background: white; font-size: 10pt; }
  header, main, footer { width: 100%; }
  header { padding-top: 0; }
  nav, .filters { display: none; }
  section { break-inside: avoid; box-shadow: none; border-radius: 0; }
  .run-table-wrap { max-height: none; overflow: visible; }
  .run-table { min-width: 0; font-size: 6.5pt; }
  thead th { position: static; }
}
"""


JS = r"""
(() => {
  const search = document.getElementById("runSearch");
  const stage = document.getElementById("stageFilter");
  const model = document.getElementById("modelFilter");
  const axis = document.getElementById("axisFilter");
  const visible = document.getElementById("visibleRuns");
  const rows = [...document.querySelectorAll("#runTable tbody tr")];
  function apply() {
    const query = search.value.trim().toLowerCase();
    let count = 0;
    rows.forEach((row) => {
      const show =
        (!query || row.dataset.search.includes(query)) &&
        (!stage.value || row.dataset.stage === stage.value) &&
        (!model.value || row.dataset.model === model.value) &&
        (!axis.value || row.dataset.axis === axis.value);
      row.hidden = !show;
      if (show) count += 1;
    });
    visible.textContent = `${count} run${count === 1 ? "" : "s"}`;
  }
  [search, stage, model, axis].forEach((control) => {
    control.addEventListener("input", apply);
    control.addEventListener("change", apply);
  });
})();
"""


def build_html(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    primary = summary["primary_result"]
    prepared = summary["prepared_data"]
    stage_counts = summary["stage_counts"]
    model_counts = summary["model_counts"]
    artifacts = summary["artifact_summary"]
    loss_image = image_data_uri(FINAL_ANALYSIS / "figures/loss_by_model_mode.png")
    gain_image = image_data_uri(FINAL_ANALYSIS / "figures/spatial_gain_ci.png")
    graph_image = image_data_uri(
        FINAL_ANALYSIS / "figures/graph_qc_degree_distance.png"
    )
    stage_bars = bar_rows(stage_counts, STAGE_LABELS)
    axis_bars = bar_rows(summary["study_axis_counts"], {})

    artifact_rows = [
        [
            item["kind"].replace("_", " "),
            item["count"],
            human_bytes(item["size_bytes"]),
        ]
        for item in artifacts["groups"]
    ]
    artifact_table = table(
        ["Artifact kind", "Records", "Registered bytes"],
        artifact_rows,
        caption=(
            "Registry sizes. Restricted predictions are summarized only by "
            "count and bytes; no row-level contents enter this report."
        ),
    )
    covariates = ", ".join(prepared["model_covariate_names"])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>BAGM registered-run audit</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <div class="eyebrow">Bio-Architectural Graph Modeling · retrospective audit</div>
  <h1>What the 158 completed runs actually show</h1>
  <p class="lede">A full inventory of model design, data, training, performance,
  experimental decisions, and the maximum defensible conclusion from the
  normal-core spatial masked-expression benchmark.</p>
  <p class="meta">Generated {h(summary["generated_at_utc"])} · Exploratory
  retrospective synthesis · No raw or row-level clinical data included</p>
  <nav aria-label="Report sections">
    <a href="#verdict">Verdict</a><a href="#inventory">Run inventory</a>
    <a href="#data">Data & training</a><a href="#models">Models</a>
    <a href="#selection">Selection</a><a href="#performance">Performance</a>
    <a href="#hypotheses">Hypotheses</a><a href="#next">Next experiment</a>
    <a href="#appendix">All runs</a>
  </nav>
</header>
<main>
<section id="verdict" class="verdict">
  <div class="eyebrow">Bottom line</div>
  <h2>The prespecified hypothesis was not supported.</h2>
  <p>Locked true-graph G1 reduced whole-node masked Huber loss from
  <strong>{fmt(primary["baseline_loss"])}</strong> to
  <strong>{fmt(primary["g1_loss"])}</strong>. The paired gain was
  <strong>{fmt(primary["delta"])}</strong>, or
  <strong>{pct(primary["relative_gain_percent"])}</strong>, with a
  spatial-block 95% interval of
  [{fmt(primary["ci_lower"])}, {fmt(primary["ci_upper"])}]. The interval is
  above zero, but the effect is roughly seven times smaller than the locked
  2% minimum and therefore fails the campaign gate.</p>
  <div class="grid four">
    <div class="stat"><strong>{summary["run_count"]}</strong><span>registered completed runs</span></div>
    <div class="stat"><strong>{stage_counts["locked_final"]}</strong><span>locked final runs</span></div>
    <div class="stat"><strong>{pct(primary["relative_gain_percent"])}</strong><span>primary G1 relative gain</span></div>
    <div class="stat"><strong>1 / 4 failed</strong><span>all four gate criteria were required</span></div>
  </div>
  <dl class="facts">
    <dt>Fact</dt><dd>G1 was slightly better than B0, B1, Broad-Field, the
    parameter-matched self model, and rewired G1 on whole-node masking.</dd>
    <dt>Inference</dt><dd>There is weak, sub-threshold within-core evidence for
    a topology-specific predictive signal.</dd>
    <dt>Not established</dt><dd>Patient generalization, biological
    communication, mechanism, causality, or a biologically meaningful 75 µm
    interaction scale.</dd>
  </dl>
</section>

<section id="inventory">
  <div class="eyebrow">1 · Run inventory</div>
  <h2>A complete registry, but four different evidence tiers</h2>
  <p>All 158 registry entries are marked completed and have a checksum-verified
  best checkpoint. There are no registered failed runs. The evidence tier,
  however, is decisive: diagnostics and validation screens cannot be promoted
  into locked-test evidence.</p>
  <div class="grid two">
    <div><h3>Lifecycle</h3>{stage_bars}</div>
    <div><h3>Study axis</h3>{axis_bars}</div>
  </div>
  <div class="grid three">
    <div class="stat"><strong>{summary["checkpoint_verification_counts"]["verified"]}/158</strong><span>verified best checkpoints</span></div>
    <div class="stat"><strong>{summary["prediction_bearing_runs"]}</strong><span>prediction-bearing runs</span></div>
    <div class="stat"><strong>{summary["duplicate_checkpoint_records"]}</strong><span>records in {summary["duplicate_checkpoint_groups"]} exact-content groups</span></div>
  </div>
  <div class="callout warning"><strong>Legacy caveat.</strong> The registry
  stores fold 0 and attempt 1 as historical placeholders, but all 158 category
  records correctly mark fold and attempt as unknown. Only seed identity is
  known. The full run appendix preserves that distinction.</div>
  {artifact_table}
</section>

<section id="data">
  <div class="eyebrow">2 · Data and training</div>
  <h2>The models were trained on one core, not the full CosMx snapshot</h2>
  <div class="grid four">
    <div class="stat"><strong>{prepared["cells"]:,}</strong><span>cells in one selected true-Normal core</span></div>
    <div class="stat"><strong>{prepared["fovs"]}</strong><span>FOVs</span></div>
    <div class="stat"><strong>{prepared["genes"]:,}</strong><span>biological probe targets</span></div>
    <div class="stat"><strong>{prepared["covariates"]}</strong><span>visible morphology/imaging covariates</span></div>
  </div>
  <p>The broader local snapshot contains 407,999 cells, but this campaign used
  only the uniquely selected legacy true-Normal core. Technical probes beginning
  with <code>Negative</code> or <code>SystemControl</code> were excluded.
  The 1,000 retained targets were gene-wise standardized <code>log1p</code>
  counts using training-only statistics. Slash-combined probes remained
  indivisible measurements. Of the registered runs, 149 used
  <code>prepared_full_v1</code>; the nine diagnostics used
  <code>prepared_smoke_v1</code> and are not scientific evidence.</p>
  <div class="grid two">
    <div>
      <h3>Split and evaluation</h3>
      <ul class="tight">
        <li>{prepared["split_counts"]["train"]:,} train,
        {prepared["split_counts"]["validation"]:,} validation, and
        {prepared["split_counts"]["test"]:,} sealed-test cells.</li>
        <li>{prepared["macroblocks"]} disjoint spatial macroblocks; split graphs
        were constructed independently with no cross-split edges.</li>
        <li>{prepared["validation_mask_replicates"]} fixed validation and
        {prepared["test_mask_replicates"]} fixed test replicates for each mask
        mode.</li>
        <li>Cells and seeds are technical units. Spatial blocks are the
        within-core uncertainty units; there are no patient replicates.</li>
      </ul>
    </div>
    <div>
      <h3>Allowed and prohibited information</h3>
      <p class="small"><strong>Visible covariates:</strong> {h(covariates)}.</p>
      <p class="small"><strong>Excluded as node inputs:</strong> identifiers,
      coordinates, RNA-derived QC/library size, vendor cell types, clusters,
      expression-derived neighborhoods, and niches. Coordinates were used only
      for splitting/graph construction and in the explicitly labeled
      Broad-Field control.</p>
    </div>
  </div>
  <h3>Locked training recipe</h3>
  <ol class="pipeline">
    <li><strong>Encode.</strong> Project masked expression, an explicit
    1,000-gene mask channel, and 22 measured covariates into 512 hidden units.</li>
    <li><strong>Mask curriculum.</strong> Ten partial-gene warm-up epochs, then
    P+N+B sampling: 60% partial-gene, 30% whole-node, and 10% spatial-block
    epochs. Partial masks hide 20% of genes; node/block masks target 10% of
    eligible cells.</li>
    <li><strong>Optimize.</strong> AdamW, learning rate 3×10⁻⁴, weight decay
    10⁻⁴, Huber δ=1, gradient clipping at 1.0, dropout/edge dropout 0.1, up to
    200 epochs, patience 25, and restoration of the best validation checkpoint.</li>
    <li><strong>Execute.</strong> Exact full split graphs with no neighbor
    sampling or target-node minibatching; deterministic seeds; mixed precision
    after an FP32 equivalence audit passed.</li>
    <li><strong>Lock and test.</strong> Validation selected the standards.
    Ten conditions × seeds 0–4 opened the sealed test once and were aggregated
    by seed ensemble, fixed-mask replicate, then equal-weight spatial block.</li>
  </ol>
  <p class="meta">Historical environment: Python 3.12.13, PyTorch
  2.11.0+cu130, torch-geometric 2.7.0, CUDA 13.0, and RTX 3090 GPUs. All
  registered runs reference Git commit <code>{h(summary["provenance"]["git_commit"])}</code>
  with a recorded dirty worktree; provenance is present but the commit alone
  is not a clean reconstruction of those local changes.</p>
</section>

<section id="models">
  <div class="eyebrow">3 · Model basics and size</div>
  <h2>Six trained families, one gated-off model</h2>
  <p>Across diagnostics and screens, trainable sizes ranged from
  {summary["parameter_count_range"][0]:,} to
  {summary["parameter_count_range"][1]:,} parameters, with serialized
  checkpoints from {h(human_bytes(summary["checkpoint_size_range"][0]))} to
  {h(human_bytes(summary["checkpoint_size_range"][1]))}. The table reports the
  locked configuration; the appendix records each screened width and depth.</p>
  {render_model_catalog(summary)}
  <div class="callout"><strong>Size is not the result.</strong> Locked G1 and
  B0-matched both have 3,916,776 parameters. G1 still improved whole-node loss
  by only 0.210% relative to B0-matched, so capacity does not explain the
  entire difference, but the remaining effect is still small.</div>
  {render_resource_table(summary)}
</section>

<section id="selection">
  <div class="eyebrow">4 · Validation-only selection history</div>
  <h2>The standards were chosen sequentially before sealed testing</h2>
  {render_selection_table(summary)}
  <p>The selected standard was P+N+B masking; k=16, radius 75 µm, mutual graph;
  512 hidden units; two G1 layers; and 64-dimensional G2 edge embeddings.
  Selection differences were generally small. In particular, the top three
  graph radii differed by less than 0.00005 whole-node Huber, so 75 µm is a
  benchmark choice, not an identified biological range.</p>
  <div class="callout warning"><strong>Screening caution.</strong> The lowest
  validation number anywhere in the sequence is not a fair universal model
  ranking: dataset state, locked nuisance settings, model family, and stage
  changed sequentially. The locked test matrix below is the valid head-to-head
  comparison.</div>
</section>

<section id="performance">
  <div class="eyebrow">5 · Locked performance</div>
  <h2>No single model wins all estimands</h2>
  {render_final_loss_table(summary)}
  <ul class="tight">
    <li><strong>Partial-gene:</strong> B0 had the lowest point estimate
    ({fmt(next(x["huber_loss"] for x in summary["final_test_losses"] if x["condition"] == "b0" and x["mask_mode"] == "partial"))});
    visible genes in the same cell carried most of the signal.</li>
    <li><strong>Whole-node:</strong> G2 distance-only had the lowest point
    estimate, but its advantage over G2 true was 0.000018 in the opposite
    direction of the edge-feature hypothesis and its interval crossed zero.
    It is a numerical tie, not a robust winner.</li>
    <li><strong>Spatial block:</strong> true G1 had the lowest point estimate,
    but B0−G1 was {fmt(next(x["delta"] for x in summary["paired_test_gains"] if x["comparison"] == "B0_minus_G1" and x["mask_mode"] == "block"))}
    with an interval crossing zero.</li>
  </ul>
  <figure>
    <img src="{loss_image}" alt="Heatmap of locked test Huber losses for ten conditions across partial-gene, whole-node, and spatial-block masking. Values differ only slightly within each column.">
    <figcaption>Locked test seed-ensemble Huber loss. Absolute loss across mask
    modes is not directly comparable because each mode answers a different
    prediction question.</figcaption>
  </figure>
  {render_pairwise_table(summary)}
  <figure>
    <img src="{gain_image}" alt="Paired spatial-block gain intervals. Whole-node B0 minus G1 and rewired G1 minus true G1 are above zero; both block-mask intervals cross zero.">
    <figcaption>Primary and rewiring contrasts. The narrow whole-node intervals
    establish a reproducible within-core difference, not biological
    replication.</figcaption>
  </figure>
  <h3>Prespecified gate</h3>
  {render_gate_table(summary)}
  <figure>
    <img src="{graph_image}" alt="Selected graph quality chart showing mean degree about 13.7, isolated-cell rate about 0.041 percent, and 95th-percentile edge distance about 27.2 micrometres.">
    <figcaption>The selected graph had mean/median degree 13.71/14, 10 isolated
    cells, 24 components, and 95th-percentile edge length 27.19 µm. The 75 µm
    radius is mostly a cap; observed edges were usually much shorter.</figcaption>
  </figure>
</section>

<section id="hypotheses">
  <div class="eyebrow">6 · What was and was not learned</div>
  <h2>Evidence by hypothesis</h2>
  <div class="hypothesis negative">
    <h3>H1: local graph context produces a meaningful ≥2% whole-node gain</h3>
    <p><strong>Not supported.</strong> Observed gain was
    {pct(primary["relative_gain_percent"])}. Three other gate components passed,
    but the minimum effect criterion was mandatory.</p>
  </div>
  <div class="hypothesis support">
    <h3>A weak topology-specific within-core signal exists</h3>
    <p><strong>Descriptively supported.</strong> The B0−G1 and rewired−true G1
    whole-node intervals were above zero. G1 also beat B1, Broad-Field, and
    parameter-matched B0. The effect remains sub-threshold and is based on one
    core.</p>
  </div>
  <div class="hypothesis negative">
    <h3>Measured edge geometry adds useful information beyond topology</h3>
    <p><strong>Not supported.</strong> G2 improved over G1 by only 0.000014
    whole-node Huber (0.006%), with an interval crossing zero. Zero-edge,
    distance-only, and permuted-edge controls all lay within 0.000040 of true
    G2 and every paired interval crossed zero.</p>
  </div>
  <div class="hypothesis uncertain">
    <h3>The signal is biological rather than segmentation spillover or local copying</h3>
    <p><strong>Unresolved.</strong> Literal nearest-neighbor copying performed
    poorly and broad-field/rewired controls help, but segmentation perturbation
    and minimum-distance sensitivity were not conclusion-bearing tests. The
    rewired graph also shifted the 95th-percentile distance from 27.19 to
    45.41 µm, so topology and distance distribution were not perfectly
    separated.</p>
  </div>
  <div class="hypothesis uncertain">
    <h3>Any biological mechanism or causal relationship was proven</h3>
    <p><strong>No.</strong> Prediction in one core cannot establish a
    communication mechanism, and no independent-patient replication,
    faithfulness audit, orthogonal assay, or perturbation was performed.</p>
  </div>
</section>

<section id="next">
  <div class="eyebrow">7 · Recommended next experiment</div>
  <h2>Replicate the simple contrast across independent biological units—or stop</h2>
  <div class="callout negative"><strong>Do not enqueue the planned edge-feature
  ablation unchanged.</strong> Its G1-versus-G2 question was already tested more
  strongly in the locked matrix: 20 G2 runs covered true, zero-edge,
  distance-only, and permuted-edge conditions across five seeds each. The
  result was null at practical scale.</div>
  <ol class="pipeline">
    <li><strong>Fix the cohort contract first.</strong> Reconcile the legacy and
    revised pathology tables into a versioned policy while preserving donor
    grouping. The current tooling cannot do this canonically by changing a
    filename.</li>
    <li><strong>Choose the claim explicitly.</strong> A strict true-Normal
    replication requires new independent true-Normal cores because the current
    snapshot has only one. If the claim is broadened to adjacent-normal gastric
    tissue, the 14 adjacent-normal cores can support a donor-held-out study, but
    that is a new estimand and must be labeled as such.</li>
    <li><strong>Run the minimum discriminating ladder.</strong> Keep B0 versus
    true G1 as primary, with B1, Broad-Field, parameter-matched B0, and a
    degree-and-full-distance-matched rewired G1 as required controls. Retain the
    locked P+N+B curriculum and 512-wide two-layer G1 to avoid another broad
    search.</li>
    <li><strong>Attack the strongest remaining shortcut.</strong> Add
    prespecified segmentation/coordinate perturbations and minimum-distance or
    boundary-contact sensitivity. Treat 30/50/75 µm graphs as robustness
    analyses, not per-core test-set choices.</li>
    <li><strong>Aggregate by patient or donor.</strong> Primary endpoint:
    donor-level whole-node Huber contrast with heterogeneity and an uncertainty
    interval across independent donors. Keep block masking secondary and add
    MAE, deviance/correlation, abundance/dispersion strata, and calibration
    without collapsing them into one score.</li>
    <li><strong>Use a hard stop.</strong> Preserve the historical 2% gate unless
    a different meaningful effect is justified and locked before new outcomes
    are seen. If G1 does not replicate across donors, default to B0 and stop the
    attribution/G3 branch. Only a replicated graph-specific gain should advance
    to G3, faithfulness, null calibration, or biological readouts.</li>
  </ol>
  <div class="callout"><strong>Discriminating prediction.</strong> A genuine
  local-context signal should persist across held-out donors, remain positive
  after segmentation perturbation, and beat a control matched on the full edge
  distance distribution. Spillover or regional confounding predicts collapse
  under those tests.</div>
</section>

<section id="audit">
  <div class="eyebrow">8 · Integrity and limitations</div>
  <h2>What was verified</h2>
  <ul class="tight">
    <li>158 unique registry runs, 158 verified checkpoints, and 530 registered
    artifact records totaling {h(human_bytes(artifacts["total"]["size_bytes"]))}.</li>
    <li>Checkpoint state dictionaries were read with
    <code>weights_only=True</code> to reconstruct model sizes; no checkpoint
    code was executed.</li>
    <li>Exact checkpoint duplication affects
    {summary["duplicate_checkpoint_records"]} exploratory records in
    {summary["duplicate_checkpoint_groups"]} groups and reflects reused
    sequential screen results, not additional independent evidence.</li>
    <li>Aggregate recorded run time was
    {summary["aggregate_recorded_run_seconds"] / 3600:.2f} job-hours; jobs ran
    in parallel, so this is not elapsed wall-clock time.</li>
    <li>The historical final audit independently checked all 50 locked runs,
    1,200 prediction records, 2,700 array headers, prediction/checkpoint/metric
    hashes, and 119 automated tests.</li>
  </ul>
  <div class="callout warning"><strong>Remaining provenance limitation.</strong>
  Every imported run records the same Git commit and a dirty worktree. File
  hashes and immutable bundles support integrity, but exact source
  reconstruction requires the preserved dirty-state record; a commit hash
  alone is insufficient.</div>
  <p class="small"><strong>Primary local sources:</strong>
  <code>{h(summary["provenance"]["database"])}</code>,
  <code>{h(summary["provenance"]["prepared_manifest"])}</code>,
  <code>{h(summary["provenance"]["standards_lock"])}</code>, and
  <code>{h(summary["provenance"]["final_summary"])}</code>. Their SHA-256
  digests are recorded in the adjacent <code>summary.json</code>.</p>
</section>

<section id="appendix">
  <div class="eyebrow">Appendix · every registered run</div>
  <h2>Searchable 158-run table</h2>
  <p>The table exposes every run rather than only favorable seeds. Diagnostic,
  exploratory, confirmation, and locked-final evidence remain distinguishable.
  A machine-readable copy is provided as <code>runs.csv</code>.</p>
  {render_run_table(rows)}
</section>
</main>
<footer>
  <p>Portable single-file report: all figures, styles, scripts, and the complete
  run table are embedded. This report is a retrospective analysis and does not
  alter the immutable run artifacts or reopen the sealed test data.</p>
</footer>
<script>{JS}</script>
</body>
</html>
"""


def write_html(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> Path:
    path = REPORT_DIR / "run_report.html"
    path.write_text(build_html(rows, summary), encoding="utf-8")
    return path


def validate(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    csv_path: Path,
    summary_path: Path,
    html_path: Path,
) -> None:
    assert len(rows) == 158
    assert len({row["run_id"] for row in rows}) == 158
    assert Counter(row["status"] for row in rows) == {"completed": 158}
    assert Counter(row["verification_status"] for row in rows) == {
        "verified": 158
    }
    assert summary["stage_counts"] == {
        "diagnostic": 9,
        "exploratory_screen": 96,
        "locked_final": 50,
        "validation_confirmation": 3,
    }
    assert summary["model_counts"] == {
        "b0": 16,
        "b0-matched": 5,
        "b1": 24,
        "broad-field": 7,
        "g1": 76,
        "g2": 30,
    }
    assert all(int(row["parameter_count"]) > 0 for row in rows)
    assert all(
        resolved_artifact_path(str(row["checkpoint_path"])).is_file()
        for row in rows
    )
    locked = [row for row in rows if row["lifecycle_stage"] == "locked_final"]
    assert len(locked) == 50
    locked_conditions = Counter(row["condition"] for row in locked)
    assert locked_conditions == Counter({condition: 5 for condition in CONDITION_ORDER})
    for condition in CONDITION_ORDER:
        seeds = {
            int(row["seed"])
            for row in locked
            if row["condition"] == condition
        }
        assert seeds == {0, 1, 2, 3, 4}
    g1_params = {
        int(row["parameter_count"])
        for row in locked
        if row["condition"] == "g1_true"
    }
    matched_params = {
        int(row["parameter_count"])
        for row in locked
        if row["condition"] == "b0_parameter_matched"
    }
    assert g1_params == matched_params == {3_916_776}
    assert len(summary["final_test_losses"]) == 30
    failed_scientific_gates = [
        row
        for row in summary["acceptance_gate"]
        if row["category"] == "hypothesis_gate"
        and row["passed"].lower() != "true"
    ]
    assert len(failed_scientific_gates) == 1
    assert (
        failed_scientific_gates[0]["criterion"]
        == "whole_node_relative_gain_at_least_2pct"
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 158
    assert read_json(summary_path)["run_count"] == 158
    report = html_path.read_text(encoding="utf-8")
    assert report.startswith("<!doctype html>")
    assert report.count('data-run="1"') == 158
    assert report.count("data:image/png;base64,") == 3
    assert 'src="http' not in report
    assert "<script src=" not in report
    assert "<link " not in report
    assert "The prespecified hypothesis was not supported." in report
    assert "No raw or row-level clinical data included" in report
    assert html_path.stat().st_size > 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run strict reconciliation and portability checks after writing.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    rows = load_run_rows()
    csv_path = write_runs_csv(rows)
    summary = make_summary(rows)
    summary_path = write_summary(summary)
    html_path = write_html(rows, summary)
    if arguments.check:
        validate(rows, summary, csv_path, summary_path, html_path)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "runs": len(rows),
                    "html": str(html_path),
                    "html_bytes": html_path.stat().st_size,
                    "csv": str(csv_path),
                    "summary": str(summary_path),
                },
                indent=2,
            )
        )
    else:
        print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

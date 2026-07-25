"""Validation-only graph-by-mask interaction diagnostics.

This module deliberately does not choose a graph or masking curriculum.  It
verifies a complete paired 2 x 2 validation screen and reports a descriptive
difference-in-differences after averaging fixed-mask technical replicates
within each model seed.  The sealed test split is never accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .artifacts import sha256_file
from .experiment import load_run_manifest


INTERACTION_ARTIFACT_FORMAT_VERSION = 1
INTERACTION_ARTIFACT_KIND = (
    "validation_only_graph_mask_interaction_diagnostic"
)
RUN_ARTIFACT_KIND = "normal_true_tissue_spatial_benchmark_run"
GRAPH_FACTOR_FIELDS = (
    "k",
    "radius_um",
    "symmetry",
    "min_distance_um",
)
MASK_MODES = ("node", "block")
_MANIFEST_FILENAME = "manifest.json"
_CHECKSUM_FILENAME = "checksums.sha256"


class InteractionSummaryError(ValueError):
    """Raised when the interaction screen violates its paired design."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_content_hash(manifest: Mapping[str, Any]) -> str:
    core = deepcopy(dict(manifest))
    core.pop("manifest_content_sha256", None)
    return _canonical_hash(core)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _graph_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    graph = manifest.get("graph")
    config = manifest.get("config")
    if not isinstance(graph, Mapping) or not isinstance(config, Mapping):
        raise InteractionSummaryError(
            "Run manifest lacks graph or configuration declarations."
        )
    graph_config = graph.get("config")
    configured_graph = config.get("graph")
    if not isinstance(graph_config, Mapping) or not isinstance(
        configured_graph, Mapping
    ):
        raise InteractionSummaryError(
            "Run manifest lacks graph candidate configuration."
        )
    missing = [
        name
        for name in GRAPH_FACTOR_FIELDS
        if name not in graph_config
    ]
    if missing:
        raise InteractionSummaryError(
            "Graph candidate lacks required fields: "
            + ", ".join(missing)
        )
    fields = {
        name: graph_config[name]
        for name in GRAPH_FACTOR_FIELDS
    }
    if any(configured_graph.get(name) != value for name, value in fields.items()):
        raise InteractionSummaryError(
            "Graph candidate fields disagree between run declarations."
        )
    return fields


def _graph_candidate_id(fields: Mapping[str, Any]) -> str:
    digest = _canonical_hash(dict(fields))[:8]
    radius = str(fields["radius_um"]).replace(".", "p")
    minimum = str(fields["min_distance_um"]).replace(".", "p")
    return (
        f"k{fields['k']}_r{radius}_{fields['symmetry']}_"
        f"min{minimum}_{digest}"
    )


def _curriculum(manifest: Mapping[str, Any]) -> str:
    try:
        value = manifest["config"]["training"]["curriculum"]
    except (KeyError, TypeError) as exc:
        raise InteractionSummaryError(
            "Run manifest lacks its training curriculum."
        ) from exc
    if not isinstance(value, str) or not value:
        raise InteractionSummaryError(
            "Training curriculum must be a non-empty string."
        )
    return value


def _nuisance_context(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return every configured setting except the two interaction factors."""

    config = deepcopy(manifest.get("config"))
    if not isinstance(config, dict):
        raise InteractionSummaryError(
            "Run manifest configuration must be a mapping."
        )
    for section in ("run", "training", "graph"):
        if not isinstance(config.get(section), dict):
            raise InteractionSummaryError(
                f"Run configuration section {section!r} must be a mapping."
            )
    config["run"].pop("model_seed", None)
    config["training"].pop("model_seed", None)
    config["training"].pop("curriculum", None)
    for name in GRAPH_FACTOR_FIELDS:
        config["graph"].pop(name, None)
    graph = manifest.get("graph")
    if not isinstance(graph, Mapping):
        raise InteractionSummaryError(
            "Run graph declaration must be a mapping."
        )
    rewire = graph.get("rewire")
    if rewire is not None and not isinstance(rewire, Mapping):
        raise InteractionSummaryError(
            "Run rewiring declaration must be a mapping or null."
        )
    rewire_settings = (
        {
            str(key): deepcopy(value)
            for key, value in rewire.items()
            if key != "achieved"
        }
        if isinstance(rewire, Mapping)
        else None
    )
    return {
        "model_name": manifest.get("model_name"),
        "graph_kind": graph.get("kind"),
        "edge_control": graph.get("edge_control"),
        "edge_control_seed": graph.get("edge_control_seed"),
        "edge_control_contract": graph.get("edge_control_contract"),
        "rewire_settings": rewire_settings,
        "rewired": config["run"].get("rewired"),
        "configuration": config,
    }


def _prepared_provenance(
    manifest: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    prepared = manifest.get("prepared_artifact")
    if not isinstance(prepared, Mapping):
        raise InteractionSummaryError(
            "Run manifest lacks prepared-artifact provenance."
        )
    names = (
        "artifact_id",
        "manifest_sha256",
        "split_id",
        "validation_mask_bundle_id",
    )
    values = tuple(prepared.get(name) for name in names)
    if any(not isinstance(value, str) or not value for value in values):
        raise InteractionSummaryError(
            "Prepared artifact, split, or validation-mask provenance is invalid."
        )
    return values  # type: ignore[return-value]


def _metric_records(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    manifest_path: Path,
    candidate_id: str,
    graph_fields: Mapping[str, Any],
    curriculum: str,
    context_id: str,
) -> list[dict[str, Any]]:
    if metrics.get("test_targets_evaluated") is not False or metrics.get(
        "test"
    ):
        raise InteractionSummaryError(
            f"Interaction diagnostic refuses test metrics: {manifest_path}"
        )
    evaluations = metrics.get("validation")
    if not isinstance(evaluations, list) or not evaluations:
        raise InteractionSummaryError(
            f"Run has no validation evaluations: {manifest_path}"
        )
    seen: set[tuple[str, int]] = set()
    records: list[dict[str, Any]] = []
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise InteractionSummaryError(
                f"Invalid validation metric record: {manifest_path}"
            )
        mode = evaluation.get("mask_mode")
        if mode not in MASK_MODES:
            continue
        replicate = evaluation.get("mask_replicate")
        if (
            not isinstance(replicate, int)
            or isinstance(replicate, bool)
            or replicate < 0
        ):
            raise InteractionSummaryError(
                f"Invalid validation mask replicate: {manifest_path}"
            )
        key = (str(mode), replicate)
        if key in seen:
            raise InteractionSummaryError(
                "Duplicate mask-mode/replicate evaluation in run: "
                f"{manifest_path}"
            )
        seen.add(key)
        values = evaluation.get("metrics")
        if not isinstance(values, Mapping):
            raise InteractionSummaryError(
                f"Validation metrics are missing: {manifest_path}"
            )
        try:
            huber = float(values["huber"])
            n_masked = int(values["n_masked"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InteractionSummaryError(
                f"Invalid validation Huber metrics: {manifest_path}"
            ) from exc
        if not math.isfinite(huber) or huber < 0 or n_masked <= 0:
            raise InteractionSummaryError(
                f"Invalid validation Huber metrics: {manifest_path}"
            )
        records.append(
            {
                "run_id": manifest["run_id"],
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "model": manifest["model_name"],
                "model_seed": int(manifest["model_seed"]),
                "context_id": context_id,
                "graph_candidate_id": candidate_id,
                "graph_id": manifest["graph"]["graph_id"],
                **dict(graph_fields),
                "curriculum": curriculum,
                "mask_mode": str(mode),
                "mask_replicate": replicate,
                "huber": huber,
                "n_masked": n_masked,
            }
        )
    present = {record["mask_mode"] for record in records}
    if present != set(MASK_MODES):
        raise InteractionSummaryError(
            "Every run must evaluate both node and block validation masks: "
            f"{manifest_path}"
        )
    return records


def load_interaction_runs(
    root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate one complete paired 2 x 2 interaction screen."""

    supplied_root = Path(root)
    if supplied_root.is_symlink():
        raise InteractionSummaryError(
            "Interaction run directory cannot be a symbolic link."
        )
    run_root = supplied_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(
            f"Interaction run directory was not found: {run_root}"
        )
    records: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    run_cells: set[tuple[str, str, int]] = set()
    provenance: set[tuple[str, str, str, str]] = set()
    contexts: dict[str, list[str]] = {}
    graph_ids: dict[str, set[str]] = {}
    discovered = 0
    for manifest_path in sorted(run_root.rglob(_MANIFEST_FILENAME)):
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise InteractionSummaryError(
                f"Run manifest is not a regular file: {manifest_path}"
            )
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InteractionSummaryError(
                f"Invalid JSON manifest under interaction runs: {manifest_path}"
            ) from exc
        if not isinstance(raw, Mapping) or raw.get(
            "artifact_kind"
        ) != RUN_ARTIFACT_KIND:
            continue
        discovered += 1
        manifest = load_run_manifest(manifest_path.parent)
        if manifest.get("sealed_test_opened") is not False:
            raise InteractionSummaryError(
                "Interaction diagnostic accepts sealed validation runs only: "
                f"{manifest_path}"
            )
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or run_id in run_ids:
            raise InteractionSummaryError(
                f"Duplicate immutable run_id in interaction screen: {run_id}"
            )
        run_ids.add(run_id)
        prepared = _prepared_provenance(manifest)
        provenance.add(prepared)
        fields = _graph_fields(manifest)
        candidate_id = _graph_candidate_id(fields)
        curriculum = _curriculum(manifest)
        try:
            model_seed = int(manifest["model_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InteractionSummaryError(
                f"Invalid model seed: {manifest_path}"
            ) from exc
        run_cell = (candidate_id, curriculum, model_seed)
        if run_cell in run_cells:
            raise InteractionSummaryError(
                "Duplicate graph/curriculum/seed run in interaction screen: "
                f"{manifest_path}"
            )
        run_cells.add(run_cell)
        context = _nuisance_context(manifest)
        context_id = _canonical_hash(context)
        contexts.setdefault(context_id, []).append(str(manifest_path))
        graph_ids.setdefault(candidate_id, set()).add(
            str(manifest["graph"]["graph_id"])
        )
        metrics_name = manifest.get("metrics_file")
        if not isinstance(metrics_name, str):
            raise InteractionSummaryError(
                f"Run metrics filename is invalid: {manifest_path}"
            )
        metrics_path = manifest_path.parent / metrics_name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, Mapping):
            raise InteractionSummaryError(
                f"Run metrics root is invalid: {metrics_path}"
            )
        records.extend(
            _metric_records(
                manifest,
                metrics,
                manifest_path=manifest_path,
                candidate_id=candidate_id,
                graph_fields=fields,
                curriculum=curriculum,
                context_id=context_id,
            )
        )
    if discovered == 0 or not records:
        raise InteractionSummaryError(
            f"No complete benchmark runs were found under {run_root}"
        )
    if len(provenance) != 1:
        raise InteractionSummaryError(
            "Interaction runs do not share exactly one prepared artifact, "
            "split, and validation mask bundle."
        )
    if len(contexts) != 1:
        paths = sorted(path for values in contexts.values() for path in values)
        raise InteractionSummaryError(
            "Interaction runs differ in nuisance settings other than graph "
            "candidate and curriculum: "
            + ", ".join(paths)
        )
    if any(len(values) != 1 for values in graph_ids.values()):
        raise InteractionSummaryError(
            "One graph candidate maps to multiple graph IDs."
        )
    frame = pd.DataFrame.from_records(records)
    graph_candidates = sorted(frame["graph_candidate_id"].unique())
    curricula = sorted(frame["curriculum"].unique())
    if len(graph_candidates) != 2 or len(curricula) != 2:
        raise InteractionSummaryError(
            "Interaction diagnostic requires exactly two graph candidates "
            "and two curricula."
        )
    expected_cells = {
        (graph, curriculum)
        for graph in graph_candidates
        for curriculum in curricula
    }
    observed_cells = set(
        frame[["graph_candidate_id", "curriculum"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise InteractionSummaryError(
            "Interaction runs do not form a complete 2 x 2 factorial."
        )
    seed_sets = {
        (graph, curriculum): tuple(
            sorted(
                int(value)
                for value in group["model_seed"].astype(int).unique()
            )
        )
        for (graph, curriculum), group in frame.groupby(
            ["graph_candidate_id", "curriculum"],
            sort=True,
        )
    }
    if len(set(seed_sets.values())) != 1:
        raise InteractionSummaryError(
            "Every graph/curriculum cell must use the exact identical paired "
            "model-seed set: "
            + json.dumps(
                {
                    f"{graph}|{curriculum}": list(seeds)
                    for (graph, curriculum), seeds in seed_sets.items()
                },
                sort_keys=True,
            )
        )
    paired_seeds = next(iter(seed_sets.values()))
    if not paired_seeds:
        raise InteractionSummaryError(
            "Interaction screen has no paired model seeds."
        )
    replicate_sets = {
        (graph, curriculum, int(seed), mode): tuple(
            sorted(
                int(value)
                for value in group["mask_replicate"].astype(int).unique()
            )
        )
        for (graph, curriculum, seed, mode), group in frame.groupby(
            [
                "graph_candidate_id",
                "curriculum",
                "model_seed",
                "mask_mode",
            ],
            sort=True,
        )
    }
    by_mode = {
        mode: {
            replicates
            for key, replicates in replicate_sets.items()
            if key[-1] == mode
        }
        for mode in MASK_MODES
    }
    if any(len(values) != 1 for values in by_mode.values()):
        raise InteractionSummaryError(
            "Paired runs must use exact identical validation mask-replicate "
            "sets within each mask mode."
        )
    prepared = next(iter(provenance))
    metadata = {
        "run_root": str(run_root),
        "prepared_artifact_id": prepared[0],
        "prepared_manifest_sha256": prepared[1],
        "split_id": prepared[2],
        "validation_mask_bundle_id": prepared[3],
        "nuisance_context_sha256": next(iter(contexts)),
        "paired_seeds": list(paired_seeds),
        "mask_replicates": {
            mode: list(next(iter(by_mode[mode])))
            for mode in MASK_MODES
        },
        "graph_candidates": graph_candidates,
        "curricula": curricula,
        "run_ids": sorted(run_ids),
        "run_manifest_checksums": sorted(
            frame["manifest_sha256"].unique()
        ),
    }
    return frame, metadata


def aggregate_interaction(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average technical masks within seed, then summarize seed variation."""

    group = [
        "graph_candidate_id",
        "graph_id",
        *GRAPH_FACTOR_FIELDS,
        "curriculum",
        "mask_mode",
        "model_seed",
    ]
    per_seed = (
        frame.groupby(group, dropna=False, as_index=False)
        .agg(
            huber=("huber", "mean"),
            n_mask_replicates=("mask_replicate", "nunique"),
            n_masked_total=("n_masked", "sum"),
        )
        .sort_values(
            ["mask_mode", "curriculum", "graph_candidate_id", "model_seed"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    summary = (
        per_seed.groupby(group[:-1], dropna=False, as_index=False)
        .agg(
            huber_mean=("huber", "mean"),
            huber_seed_sd=("huber", "std"),
            huber_seed_min=("huber", "min"),
            huber_seed_max=("huber", "max"),
            n_seeds=("model_seed", "nunique"),
            n_mask_replicates=("n_mask_replicates", "min"),
        )
        .sort_values(
            ["mask_mode", "curriculum", "graph_candidate_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return per_seed, summary


def difference_in_differences(
    per_seed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute paired descriptive differences in differences for both modes."""

    graph_rows = (
        per_seed[
            ["graph_candidate_id", *GRAPH_FACTOR_FIELDS]
        ]
        .drop_duplicates()
        .sort_values(list(GRAPH_FACTOR_FIELDS), kind="mergesort")
    )
    graphs = graph_rows["graph_candidate_id"].tolist()
    curricula = sorted(per_seed["curriculum"].unique())
    if len(graphs) != 2 or len(curricula) != 2:
        raise InteractionSummaryError(
            "Difference-in-differences requires a 2 x 2 screen."
        )
    graph_a, graph_b = graphs
    curriculum_a, curriculum_b = curricula
    paired_rows: list[dict[str, Any]] = []
    for mode in MASK_MODES:
        values = per_seed[per_seed["mask_mode"] == mode].pivot(
            index="model_seed",
            columns=["graph_candidate_id", "curriculum"],
            values="huber",
        )
        required = [
            (graph_a, curriculum_a),
            (graph_b, curriculum_a),
            (graph_a, curriculum_b),
            (graph_b, curriculum_b),
        ]
        if any(column not in values for column in required) or values[
            required
        ].isna().any().any():
            raise InteractionSummaryError(
                f"Incomplete paired difference-in-differences for {mode}."
            )
        for seed, row in values.sort_index().iterrows():
            effect_a = float(
                row[(graph_a, curriculum_a)]
                - row[(graph_b, curriculum_a)]
            )
            effect_b = float(
                row[(graph_a, curriculum_b)]
                - row[(graph_b, curriculum_b)]
            )
            paired_rows.append(
                {
                    "mask_mode": mode,
                    "model_seed": int(seed),
                    "graph_a": graph_a,
                    "graph_b": graph_b,
                    "curriculum_a": curriculum_a,
                    "curriculum_b": curriculum_b,
                    "graph_a_minus_b_under_curriculum_a": effect_a,
                    "graph_a_minus_b_under_curriculum_b": effect_b,
                    "difference_in_differences": effect_a - effect_b,
                }
            )
    paired = pd.DataFrame.from_records(paired_rows)
    summaries = []
    for mode, group in paired.groupby("mask_mode", sort=False):
        values = group["difference_in_differences"]
        summaries.append(
            {
                "mask_mode": str(mode),
                "n_paired_seeds": int(group["model_seed"].nunique()),
                "mean": float(values.mean()),
                "seed_sd": (
                    float(values.std())
                    if len(values) > 1
                    else None
                ),
                "min": float(values.min()),
                "max": float(values.max()),
                "per_seed": [
                    {
                        "model_seed": int(row.model_seed),
                        "graph_a_minus_b_under_curriculum_a": float(
                            row.graph_a_minus_b_under_curriculum_a
                        ),
                        "graph_a_minus_b_under_curriculum_b": float(
                            row.graph_a_minus_b_under_curriculum_b
                        ),
                        "difference_in_differences": float(
                            row.difference_in_differences
                        ),
                    }
                    for row in group.itertuples(index=False)
                ],
            }
        )
    graph_definitions = {
        str(row.graph_candidate_id): {
            name: getattr(row, name)
            for name in GRAPH_FACTOR_FIELDS
        }
        for row in graph_rows.itertuples(index=False)
    }
    diagnostic = {
        "artifact_role": "graph_by_mask_interaction_diagnostic",
        "diagnostic_only": True,
        "descriptive_only": True,
        "selection_performed": False,
        "selection_recommendation": {
            "made": False,
            "action": "none",
            "reason": (
                "This interaction check cannot select or override the "
                "separately locked graph or masking curriculum."
            ),
        },
        "locked_graph_or_mask_overridden": False,
        "validation_metrics_used": True,
        "test_metrics_used": False,
        "estimand": (
            "(Huber(graph A, curriculum A) - Huber(graph B, curriculum A)) "
            "- (Huber(graph A, curriculum B) - "
            "Huber(graph B, curriculum B))"
        ),
        "orientation": (
            "Huber loss is lower-is-better. A positive value means the "
            "graph-A minus graph-B loss gap is larger under curriculum A "
            "than under curriculum B."
        ),
        "aggregation": (
            "Fixed-mask replicates are averaged within each model seed "
            "before paired differences are calculated; seeds are reported "
            "as technical variation, not biological replicates."
        ),
        "graph_a": graph_a,
        "graph_b": graph_b,
        "graph_definitions": graph_definitions,
        "curriculum_a": curriculum_a,
        "curriculum_b": curriculum_b,
        "difference_in_differences": summaries,
    }
    return paired, diagnostic


def _plot_interaction(summary: pd.DataFrame, path: Path) -> None:
    curricula = sorted(summary["curriculum"].unique())
    graphs = (
        summary[
            ["graph_candidate_id", *GRAPH_FACTOR_FIELDS]
        ]
        .drop_duplicates()
        .sort_values(list(GRAPH_FACTOR_FIELDS), kind="mergesort")[
            "graph_candidate_id"
        ]
        .tolist()
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for axis, mode in zip(axes, MASK_MODES, strict=True):
        values = summary[summary["mask_mode"] == mode]
        for graph in graphs:
            graph_values = (
                values[values["graph_candidate_id"] == graph]
                .set_index("curriculum")
                .reindex(curricula)
            )
            y = graph_values["huber_mean"].to_numpy(dtype=float)
            error = (
                graph_values["huber_seed_sd"]
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            axis.errorbar(
                range(len(curricula)),
                y,
                yerr=error,
                marker="o",
                capsize=3,
                label=graph,
            )
        axis.set_xticks(range(len(curricula)))
        axis.set_xticklabels(curricula)
        axis.set_title(f"{mode.capitalize()} validation masks")
        axis.set_xlabel("Training masking curriculum")
        axis.set_ylabel("Validation masked Huber loss (lower is better)")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(
        title="Graph candidate",
        fontsize=7,
        title_fontsize=8,
    )
    figure.suptitle(
        "Validation-only graph x mask diagnostic; no standard selected; "
        "sealed test unused"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_checksums(root: Path, names: Sequence[str]) -> None:
    (root / _CHECKSUM_FILENAME).write_text(
        "".join(
            f"{sha256_file(root / name)}  {name}\n"
            for name in sorted(names)
        ),
        encoding="ascii",
    )


def _read_checksums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        try:
            checksum, name = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise InteractionSummaryError(
                "Invalid interaction checksum line."
            ) from exc
        if (
            len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            or not name
            or Path(name).name != name
            or name in records
            or name in {_MANIFEST_FILENAME, _CHECKSUM_FILENAME}
        ):
            raise InteractionSummaryError(
                "Invalid interaction checksum record."
            )
        records[name] = checksum
    return records


def create_interaction_summary(
    runs: str | Path,
    output: str | Path,
    *,
    command: Sequence[str] | None = None,
) -> Path:
    """Create an immutable, atomic graph-by-mask diagnostic artifact."""

    supplied_destination = Path(output)
    if supplied_destination.exists() or supplied_destination.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite interaction diagnostic: "
            f"{supplied_destination}"
        )
    destination = supplied_destination.resolve()
    if command is not None and (
        isinstance(command, (str, bytes))
        or any(not isinstance(item, str) for item in command)
    ):
        raise InteractionSummaryError(
            "command must be a sequence of strings."
        )
    frame, metadata = load_interaction_runs(runs)
    per_seed, summary = aggregate_interaction(frame)
    paired, diagnostic = difference_in_differences(per_seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        frame.to_csv(temporary / "validation_records.csv", index=False)
        per_seed.to_csv(
            temporary / "validation_per_seed.csv",
            index=False,
        )
        summary.to_csv(
            temporary / "validation_summary.csv",
            index=False,
        )
        paired.to_csv(
            temporary / "paired_difference_in_differences.csv",
            index=False,
        )
        diagnostic_value = {
            **diagnostic,
            "prepared_artifact_id": metadata["prepared_artifact_id"],
            "split_id": metadata["split_id"],
            "validation_mask_bundle_id": metadata[
                "validation_mask_bundle_id"
            ],
            "paired_seeds": metadata["paired_seeds"],
            "mask_replicates": metadata["mask_replicates"],
        }
        _write_json(temporary / "diagnostic.json", diagnostic_value)
        _plot_interaction(
            summary,
            temporary / "validation_interaction.png",
        )
        data_names = [
            path.name
            for path in temporary.iterdir()
            if path.is_file()
        ]
        _write_checksums(temporary, data_names)
        file_names = data_names + [_CHECKSUM_FILENAME]
        manifest: dict[str, Any] = {
            "format_version": INTERACTION_ARTIFACT_FORMAT_VERSION,
            "artifact_kind": INTERACTION_ARTIFACT_KIND,
            "status": "complete",
            "diagnostic_only": True,
            "descriptive_only": True,
            "selection_performed": False,
            "locked_graph_or_mask_overridden": False,
            "selection_scope": "validation_only",
            "test_metrics_used": False,
            "prepared_artifact": {
                "artifact_id": metadata["prepared_artifact_id"],
                "manifest_sha256": metadata[
                    "prepared_manifest_sha256"
                ],
                "split_id": metadata["split_id"],
                "validation_mask_bundle_id": metadata[
                    "validation_mask_bundle_id"
                ],
            },
            "paired_seeds": metadata["paired_seeds"],
            "mask_replicates": metadata["mask_replicates"],
            "graph_candidates": metadata["graph_candidates"],
            "curricula": metadata["curricula"],
            "nuisance_context_sha256": metadata[
                "nuisance_context_sha256"
            ],
            "run_ids": metadata["run_ids"],
            "run_manifest_checksums": metadata[
                "run_manifest_checksums"
            ],
            "provenance": {
                "runs": metadata["run_root"],
                "command": list(command or []),
            },
            "files": {
                name: sha256_file(temporary / name)
                for name in sorted(file_names)
            },
        }
        manifest["artifact_id"] = _canonical_hash(manifest)[:16]
        manifest["manifest_content_sha256"] = _manifest_content_hash(
            manifest
        )
        _write_json(temporary / _MANIFEST_FILENAME, manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_interaction_summary(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and checksum-verify a completed interaction diagnostic."""

    root = Path(path)
    manifest_path = root / _MANIFEST_FILENAME
    if root.is_symlink() or not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Interaction diagnostic artifact was not found: {root}"
        )
    if manifest_path.is_symlink():
        raise InteractionSummaryError(
            "Interaction manifest cannot be a symbolic link."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("format_version")
        != INTERACTION_ARTIFACT_FORMAT_VERSION
        or manifest.get("artifact_kind") != INTERACTION_ARTIFACT_KIND
        or manifest.get("status") != "complete"
        or manifest.get("diagnostic_only") is not True
        or manifest.get("selection_performed") is not False
        or manifest.get("locked_graph_or_mask_overridden") is not False
        or manifest.get("test_metrics_used") is not False
    ):
        raise InteractionSummaryError(
            "Unsupported or non-diagnostic interaction manifest."
        )
    if manifest.get(
        "manifest_content_sha256"
    ) != _manifest_content_hash(manifest):
        raise InteractionSummaryError(
            "Interaction manifest content checksum mismatch."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or _CHECKSUM_FILENAME not in files:
        raise InteractionSummaryError(
            "Interaction artifact checksum declarations are missing."
        )
    entries = list(root.iterdir())
    if any(
        item.is_symlink() or not item.is_file()
        for item in entries
    ):
        raise InteractionSummaryError(
            "Interaction artifact contains a directory or symbolic link."
        )
    actual = {
        item.name: item
        for item in entries
        if item.name != _MANIFEST_FILENAME
    }
    if set(actual) != set(files):
        raise InteractionSummaryError(
            "Interaction artifact file set differs from its manifest."
        )
    for name, item in actual.items():
        if sha256_file(item) != files[name]:
            raise InteractionSummaryError(
                f"Interaction artifact checksum mismatch for {name}."
            )
    checksum_records = _read_checksums(root / _CHECKSUM_FILENAME)
    expected_checksum_names = set(actual).difference({_CHECKSUM_FILENAME})
    if set(checksum_records) != expected_checksum_names:
        raise InteractionSummaryError(
            "Interaction checksum file has the wrong file set."
        )
    for name, checksum in checksum_records.items():
        if sha256_file(root / name) != checksum:
            raise InteractionSummaryError(
                f"Interaction checksum mismatch for {name}."
            )
    diagnostic = json.loads(
        (root / "diagnostic.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(diagnostic, Mapping)
        or diagnostic.get("diagnostic_only") is not True
        or diagnostic.get("selection_performed") is not False
        or diagnostic.get("locked_graph_or_mask_overridden") is not False
        or diagnostic.get("test_metrics_used") is not False
    ):
        raise InteractionSummaryError(
            "Interaction diagnostic JSON violates its non-selection contract."
        )
    return dict(manifest), dict(diagnostic)

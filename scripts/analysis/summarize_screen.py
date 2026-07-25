#!/usr/bin/env python3
"""Summarize validation-only screening runs and lock a standard."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.artifacts import sha256_file  # noqa: E402
from spatial_benchmark.experiment import load_run_manifest  # noqa: E402


SELECTION_MODEL = {
    "graph": "g1",
    "mask": "g1",
    "hidden": "g1",
    "depth": "g1",
    "edge_embedding": "g2",
    "representation": "g1",
}


def _candidate_fields(manifest: dict[str, Any], selection: str) -> dict[str, Any]:
    graph = manifest["graph"]
    config = manifest["config"]
    if selection == "graph":
        return {
            "k": graph["config"]["k"],
            "radius_um": graph["config"]["radius_um"],
            "symmetry": graph["config"]["symmetry"],
            "min_distance_um": graph["config"].get("min_distance_um", 0.0),
        }
    if selection == "mask":
        return {"curriculum": config["training"]["curriculum"]}
    if selection == "hidden":
        return {"hidden_dim": config["model"]["hidden_dim"]}
    if selection == "depth":
        return {"graph_layers": config["model"].get("graph_layers", 1)}
    if selection == "edge_embedding":
        return {
            "edge_embedding_dim": config["model"]["edge_embedding_dim"],
        }
    if selection == "representation":
        model = config["model"]
        return {
            "hidden_dim": model["hidden_dim"],
            "graph_layers": model.get("graph_layers", 1),
            "edge_embedding_dim": model.get("edge_embedding_dim"),
        }
    raise ValueError(
        "selection must be graph, mask, hidden, depth, edge_embedding, "
        "or representation"
    )


def _candidate_id(fields: dict[str, Any]) -> str:
    text = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    readable = "_".join(f"{key}-{value}" for key, value in fields.items())
    return f"{readable}_{digest}".replace(".", "p").replace("+", "")


def _comparison_context(
    manifest: dict[str, Any],
    selection: str,
) -> dict[str, Any]:
    """Return nuisance settings that must be identical across candidates."""

    config = copy.deepcopy(manifest["config"])
    config["run"].pop("model_seed", None)
    config["training"].pop("model_seed", None)
    config["model"].pop("name", None)
    if selection == "graph":
        for name in ("k", "radius_um", "symmetry", "min_distance_um"):
            config["graph"].pop(name, None)
    elif selection == "mask":
        config["training"].pop("curriculum", None)
    elif selection == "hidden":
        config["model"].pop("hidden_dim", None)
    elif selection == "depth":
        config["model"].pop("graph_layers", None)
    elif selection == "edge_embedding":
        config["model"].pop("edge_embedding_dim", None)
    elif selection == "representation":
        for name in ("hidden_dim", "graph_layers", "edge_embedding_dim"):
            config["model"].pop(name, None)
    prepared = manifest["prepared_artifact"]
    return {
        "prepared_artifact_id": prepared["artifact_id"],
        "prepared_manifest_sha256": prepared["manifest_sha256"],
        "split_id": prepared["split_id"],
        "validation_mask_bundle_id": prepared[
            "validation_mask_bundle_id"
        ],
        "model": manifest["model_name"],
        "graph_kind": manifest["graph"]["kind"],
        "edge_control": manifest["graph"]["edge_control"],
        "config": config,
    }


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(root: Path, selection: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    contexts: dict[str, list[tuple[str, str]]] = {}
    provenance: set[tuple[str, str, str, str]] = set()
    run_ids: set[str] = set()
    for manifest_path in sorted(root.rglob("manifest.json")):
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if raw.get("status") != "complete" or "model_name" not in raw:
            continue
        manifest = load_run_manifest(manifest_path.parent)
        if manifest["sealed_test_opened"] is not False:
            raise ValueError(
                f"Screening summary refuses an opened-test run: {manifest_path}"
            )
        if manifest["run_id"] in run_ids:
            raise ValueError(
                f"Duplicate immutable run_id in screen: {manifest['run_id']}"
            )
        run_ids.add(manifest["run_id"])
        metrics_path = manifest_path.parent / manifest.get(
            "metrics_file", "metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("test") or metrics.get("test_targets_evaluated") is not False:
            raise ValueError(
                f"Screening summary refuses a run with opened test metrics: {manifest_path}"
            )
        prepared = manifest["prepared_artifact"]
        provenance.add(
            (
                prepared["artifact_id"],
                prepared["manifest_sha256"],
                prepared["split_id"],
                prepared["validation_mask_bundle_id"],
            )
        )
        context = _comparison_context(manifest, selection)
        context_id = _canonical_hash(context)
        contexts.setdefault(manifest["model_name"], []).append(
            (context_id, str(manifest_path))
        )
        fields = _candidate_fields(manifest, selection)
        candidate_id = _candidate_id(fields)
        for evaluation in metrics.get("validation", []):
            records.append(
                {
                    "run_id": manifest["run_id"],
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "model": manifest["model_name"],
                    "model_seed": int(manifest["model_seed"]),
                    "context_id": context_id,
                    "prepared_artifact_id": prepared["artifact_id"],
                    "split_id": prepared["split_id"],
                    "validation_mask_bundle_id": prepared[
                        "validation_mask_bundle_id"
                    ],
                    "graph_id": manifest["graph"]["graph_id"],
                    "graph_kind": manifest["graph"]["kind"],
                    "candidate_id": candidate_id,
                    **fields,
                    "mask_mode": evaluation["mask_mode"],
                    "mask_replicate": int(evaluation["mask_replicate"]),
                    "huber": float(evaluation["metrics"]["huber"]),
                    "mse": float(evaluation["metrics"]["mse"]),
                    "mae": float(evaluation["metrics"]["mae"]),
                    "n_masked": int(evaluation["metrics"]["n_masked"]),
                    "best_epoch": int(manifest["training"]["best_epoch"]),
                    "runtime_seconds": float(manifest["timing"]["runtime_seconds"]),
                    "peak_cuda_memory_bytes": int(
                        manifest["resources"]["peak_cuda_memory_bytes"]
                    ),
                }
            )
    if not records:
        raise ValueError(f"No complete validation screening records under {root}")
    if len(provenance) != 1:
        raise ValueError(
            "Screen runs do not share one prepared artifact, split, and "
            "validation mask bundle."
        )
    selection_model = SELECTION_MODEL[selection]
    model_contexts = contexts.get(selection_model, [])
    if not model_contexts:
        raise ValueError(
            f"Screen has no comparison-model runs for {selection_model}."
        )
    unique_contexts = {context_id for context_id, _ in model_contexts}
    if len(unique_contexts) != 1:
        paths = [path for _, path in model_contexts]
        raise ValueError(
            "Comparison candidates differ in fixed nuisance settings: "
            + ", ".join(paths)
        )
    frame = pd.DataFrame.from_records(records)
    duplicate_columns = [
        "model",
        "candidate_id",
        "mask_mode",
        "mask_replicate",
        "model_seed",
    ]
    duplicate = frame.duplicated(duplicate_columns, keep=False)
    if duplicate.any():
        paths = sorted(frame.loc[duplicate, "manifest_path"].unique())
        raise ValueError(
            "Duplicate candidate/model/seed/mask evaluations: "
            + ", ".join(paths)
        )
    return frame


def _attach_graph_qc(
    frame: pd.DataFrame,
    graph_qc_path: Path,
) -> pd.DataFrame:
    """Attach prespecified train/validation-only geometry eligibility."""

    manifest_path = graph_qc_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared_ids = set(frame["prepared_artifact_id"])
    split_ids = set(frame["split_id"])
    if (
        manifest.get("artifact_kind") != "geometry_only_graph_grid"
        or manifest.get("selection_scope")
        != "train_and_validation_geometry_only"
        or manifest.get("test_geometry_used_for_selection_qc") is not False
        or manifest.get("test_expression_targets_evaluated") is not False
        or {manifest.get("prepared_artifact_id")} != prepared_ids
        or {manifest.get("split_id")} != split_ids
        or manifest.get("files", {}).get(graph_qc_path.name)
        != sha256_file(graph_qc_path)
    ):
        raise ValueError("Graph-QC artifact is incompatible or unverified.")
    qc = pd.read_csv(graph_qc_path)
    required = {
        "k",
        "radius_um",
        "symmetry",
        "selection_n_nodes",
        "selection_n_isolated_nodes",
        "selection_median_degree",
        "selection_edge_distance_max_um",
        "selection_zero_distance_edges",
        "selection_self_loops",
        "selection_duplicate_directed_edges",
        "selection_cross_group_edges",
        "selection_directed_edge_pairs_are_symmetric",
    }
    if not required.issubset(qc):
        raise ValueError("Graph-QC table lacks selection-scope fields.")
    if qc.duplicated(["k", "radius_um", "symmetry"]).any():
        raise ValueError("Graph-QC table has duplicate graph candidates.")
    qc = qc.copy()
    isolated_rate = (
        qc["selection_n_isolated_nodes"] / qc["selection_n_nodes"]
    )
    qc["qc_eligible"] = (
        (qc["selection_n_nodes"] > 0)
        & (isolated_rate <= 0.01)
        & (qc["selection_median_degree"] >= 2)
        & (qc["selection_edge_distance_max_um"] <= qc["radius_um"] + 1e-4)
        & (qc["selection_zero_distance_edges"] == 0)
        & (qc["selection_self_loops"] == 0)
        & (qc["selection_duplicate_directed_edges"] == 0)
        & (qc["selection_cross_group_edges"] == 0)
        & qc["selection_directed_edge_pairs_are_symmetric"].astype(bool)
    )
    qc["qc_isolated_rate"] = isolated_rate
    keep = [
        "k",
        "radius_um",
        "symmetry",
        "qc_eligible",
        "qc_isolated_rate",
        "selection_median_degree",
        "selection_edge_distance_max_um",
    ]
    result = frame.merge(
        qc[keep],
        on=["k", "radius_um", "symmetry"],
        how="left",
        validate="many_to_one",
    )
    if result["qc_eligible"].isna().any():
        raise ValueError("Graph-QC table does not cover every screened graph.")
    return result


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    # Mask replicates are technical repeats: average them within each model seed
    # before showing seed variability.
    group = [
        "model",
        "graph_kind",
        "candidate_id",
        "mask_mode",
        "model_seed",
    ]
    per_seed = (
        frame.groupby(group, dropna=False, as_index=False)
        .agg(
            huber=("huber", "mean"),
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            n_mask_replicates=("mask_replicate", "nunique"),
            runtime_seconds=("runtime_seconds", "mean"),
            peak_cuda_memory_bytes=("peak_cuda_memory_bytes", "max"),
        )
    )
    summary = (
        per_seed.groupby(group[:-1], dropna=False, as_index=False)
        .agg(
            huber_mean=("huber", "mean"),
            huber_sd=("huber", "std"),
            mse_mean=("mse", "mean"),
            mae_mean=("mae", "mean"),
            n_seeds=("model_seed", "nunique"),
            n_mask_replicates=("n_mask_replicates", "min"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            peak_cuda_memory_bytes_max=("peak_cuda_memory_bytes", "max"),
        )
    )
    if "qc_eligible" in frame:
        qc = (
            frame.groupby("candidate_id", as_index=False)
            .agg(
                qc_eligible=("qc_eligible", "min"),
                qc_isolated_rate=("qc_isolated_rate", "first"),
                selection_median_degree=(
                    "selection_median_degree",
                    "first",
                ),
                selection_edge_distance_max_um=(
                    "selection_edge_distance_max_um",
                    "first",
                ),
            )
        )
        summary = summary.merge(
            qc,
            on="candidate_id",
            how="left",
            validate="many_to_one",
        )
    return summary


def _recommend(
    summary: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    selection: str,
    required_seeds: int,
) -> dict[str, Any]:
    if required_seeds < 1:
        raise ValueError("required_seeds must be positive")
    selection_model = SELECTION_MODEL[selection]
    comparison = frame[
        (frame["model"] == selection_model)
        & (frame["graph_kind"] == "true")
    ]
    seed_sets = {
        str(candidate): tuple(sorted(set(group["model_seed"].astype(int))))
        for candidate, group in comparison.groupby("candidate_id")
    }
    overcomplete = {
        candidate: seeds
        for candidate, seeds in seed_sets.items()
        if len(seeds) > required_seeds
    }
    if overcomplete:
        raise ValueError(
            "Candidates exceed the declared seed count: "
            + json.dumps(overcomplete, sort_keys=True)
        )
    eligible_seed_sets = {
        candidate: seeds
        for candidate, seeds in seed_sets.items()
        if len(seeds) == required_seeds
    }
    if eligible_seed_sets and len(set(eligible_seed_sets.values())) != 1:
        raise ValueError(
            "Eligible candidates do not use one identical paired seed set: "
            + json.dumps(eligible_seed_sets, sort_keys=True)
        )
    eligible_ids = set(eligible_seed_sets)
    eligible = summary[
        (summary["model"] == selection_model)
        & (summary["graph_kind"] == "true")
        & (summary["candidate_id"].isin(eligible_ids))
        & (summary["n_seeds"] == required_seeds)
        & (summary["mask_mode"].isin(["node", "block"]))
    ]
    if "qc_eligible" in eligible:
        eligible = eligible[eligible["qc_eligible"].astype(bool)]
    pivot = eligible.pivot_table(
        index="candidate_id",
        columns="mask_mode",
        values=["huber_mean", "huber_sd"],
        aggfunc="first",
    )
    pivot.columns = [
        f"{mode}_{stat.replace('huber_', '')}"
        for stat, mode in pivot.columns
    ]
    pivot = pivot.dropna(subset=["node_mean", "block_mean"])
    if pivot.empty:
        return {
            "locked": False,
            "reason": (
                f"No {selection_model.upper()} candidate has node/block "
                "metrics, geometry eligibility, and the required "
                f"{required_seeds} seeds."
            ),
            "selection": selection,
        }
    if "node_sd" not in pivot:
        pivot["node_sd"] = 0.0
    else:
        pivot["node_sd"] = pivot["node_sd"].fillna(0.0)
    pivot = pivot.assign(candidate_sort=pivot.index.astype(str))
    ranked = pivot.sort_values(
        ["node_mean", "block_mean", "node_sd", "candidate_sort"],
        kind="mergesort",
    )
    selected_id = str(ranked.index[0])
    selected_rows = frame[frame["candidate_id"] == selected_id]
    fields = {
        key: selected_rows.iloc[0][key]
        for key in (
            ("k", "radius_um", "symmetry", "min_distance_um")
            if selection == "graph"
            else (
                ("curriculum",)
                if selection == "mask"
                else (
                    ("hidden_dim",)
                    if selection == "hidden"
                    else (
                        ("graph_layers",)
                        if selection == "depth"
                        else (
                            ("edge_embedding_dim",)
                            if selection == "edge_embedding"
                            else (
                                "hidden_dim",
                                "graph_layers",
                                "edge_embedding_dim",
                            )
                        )
                    )
                )
            )
        )
    }
    fields = {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in fields.items()
    }
    decision_rule = (
        "lexicographically minimize validation whole-node mean Huber, "
        "then validation spatial-block mean Huber, then whole-node seed SD"
    )
    if selection == "graph":
        decision_rule += (
            ", among train/validation geometry-eligible candidates"
        )
    else:
        decision_rule += ", under identical locked nuisance settings"
    return {
        "locked": True,
        "selection": selection,
        "decision_rule": decision_rule,
        "required_seeds": required_seeds,
        "paired_seeds": list(eligible_seed_sets[selected_id]),
        "candidate_id": selected_id,
        "standard": fields,
        "whole_node_huber": float(ranked.iloc[0]["node_mean"]),
        "spatial_block_huber": float(ranked.iloc[0]["block_mean"]),
        "whole_node_seed_sd": float(ranked.iloc[0]["node_sd"]),
        "ranked_candidates": [
            {
                "candidate_id": str(candidate_id),
                "whole_node_huber": float(row["node_mean"]),
                "spatial_block_huber": float(row["block_mean"]),
                "whole_node_seed_sd": float(row["node_sd"]),
                "paired_seeds": list(eligible_seed_sets[str(candidate_id)]),
            }
            for candidate_id, row in ranked.iterrows()
        ],
        "test_metrics_used": False,
    }


def _plot(
    summary: pd.DataFrame,
    path: Path,
    *,
    selection: str,
) -> None:
    selection_model = SELECTION_MODEL[selection]
    values = summary[
        (summary["model"] == selection_model)
        & (summary["graph_kind"] == "true")
        & (summary["mask_mode"].isin(["node", "block"]))
    ].copy()
    pivot = values.pivot_table(
        index="candidate_id",
        columns="mask_mode",
        values="huber_mean",
        aggfunc="first",
    ).sort_values("node")
    figure, axis = plt.subplots(figsize=(max(8, 0.35 * len(pivot)), 5.5))
    positions = range(len(pivot))
    axis.plot(positions, pivot["node"], "o-", label="Whole-node (primary)")
    axis.plot(positions, pivot["block"], "s-", label="Spatial block")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(pivot.index, rotation=75, ha="right", fontsize=7)
    axis.set_ylabel("Validation masked Huber loss (lower is better)")
    axis.set_title("Validation-only standard screen; sealed test unused")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument(
        "--selection",
        required=True,
        choices=(
            "graph",
            "mask",
            "hidden",
            "depth",
            "edge_embedding",
            "representation",
        ),
    )
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument(
        "--graph-qc",
        type=Path,
        help=(
            "Required for graph selection: graph_qc.csv with "
            "train/validation-only selection fields."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite screening summary: {destination}"
        )
    if args.selection == "graph" and args.graph_qc is None:
        raise ValueError("--graph-qc is required for graph selection")
    if args.selection != "graph" and args.graph_qc is not None:
        raise ValueError("--graph-qc applies only to graph selection")
    frame = _load(args.runs, args.selection)
    if args.graph_qc is not None:
        frame = _attach_graph_qc(frame, args.graph_qc.resolve())
    summary = _aggregate(frame)
    recommendation = _recommend(
        summary,
        frame,
        selection=args.selection,
        required_seeds=args.required_seeds,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        frame.to_csv(temporary / "validation_records.csv", index=False)
        summary.to_csv(temporary / "validation_summary.csv", index=False)
        (temporary / "recommendation.json").write_text(
            json.dumps(recommendation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _plot(
            summary,
            temporary / "validation_screen.png",
            selection=args.selection,
        )
        files = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        artifact_core = {
            "format_version": 1,
            "artifact_kind": "validation_only_screen_summary",
            "status": "complete",
            "selection": args.selection,
            "required_seeds": args.required_seeds,
            "test_metrics_used": False,
            "run_manifest_checksums": sorted(
                set(frame["manifest_sha256"])
            ),
            "files": files,
        }
        artifact_core["artifact_id"] = _canonical_hash(artifact_core)[:16]
        (temporary / "manifest.json").write_text(
            json.dumps(artifact_core, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(recommendation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

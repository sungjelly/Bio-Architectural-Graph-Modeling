from __future__ import annotations

import csv
import copy
from dataclasses import replace
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.analysis import (  # noqa: E402
    AnalysisContractError,
    PredictionRecord,
    analyze_run_directory,
    ensemble_run_records,
    evaluate_acceptance_gate,
    load_run_collection,
    validate_locked_final_execution,
)
from spatial_benchmark.masking import (  # noqa: E402
    MaskSpec,
    create_fixed_mask_bundle,
    load_fixed_mask_bundle,
    save_fixed_mask_bundle,
)
from spatial_benchmark.standards_lock import (  # noqa: E402
    authorize_locked_test_job,
    condition_name,
    create_standards_lock,
    expand_matrix,
    load_standards_lock,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_content_sha256(manifest: dict[str, object]) -> str:
    core = dict(manifest)
    core.pop("manifest_content_sha256", None)
    payload = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prepared_manifest_content_sha256(manifest: dict[str, object]) -> str:
    core = copy.deepcopy(manifest)
    core.pop("artifact_id", None)
    core.pop("manifest_content_sha256", None)
    payload = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _graph_qc() -> dict[str, object]:
    return {
        "n_nodes": 16,
        "n_directed_edges": 160,
        "n_undirected_edges": 80,
        "n_components": 1,
        "n_isolated_nodes": 0,
        "mean_degree": 10.0,
        "median_degree": 10.0,
        "p95_degree": 12.0,
        "max_degree": 13,
        "edge_distance_mean_um": 21.0,
        "edge_distance_p50_um": 20.0,
        "edge_distance_p95_um": 44.0,
        "edge_distance_max_um": 49.0,
        "zero_distance_edges": 0,
        "cap_hit_rate": 0.25,
        "self_loops": 0,
        "duplicate_directed_edges": 0,
        "fov_seam_edges": 4,
        "fov_seam_fraction": 0.05,
        "cross_group_edges": 0,
        "directed_edge_pairs_are_symmetric": True,
    }


def _write_run(
    root: Path,
    *,
    model: str,
    model_seed: int,
    graph_id: str,
    graph_kind: str,
    prediction_level: float,
    base_graph_id: str | None = None,
    modes: tuple[str, ...] = ("partial", "node", "block"),
    replicates: tuple[int, ...] = (0, 1),
    sealed_test_opened: bool = True,
    save_predictions: bool = True,
    edge_control: str = "none",
    standards_lock: dict[str, object] | None = None,
    prepared_artifact: dict[str, object] | None = None,
    locked_job: dict[str, object] | None = None,
) -> Path:
    identity = (
        f"{model}|{graph_kind}|{graph_id}|{edge_control}|{model_seed}"
    )
    run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    default_target = np.zeros((16, 4), dtype=np.float32)
    default_block_ids = np.repeat(np.arange(8, dtype=np.int64), 2)
    default_cell_ids = np.arange(16, dtype=np.int64)
    canonical_views: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    canonical_bundles: dict[str, object] = {}
    if prepared_artifact is not None:
        prepared_root = Path(str(prepared_artifact["path"]))
        with np.load(
            prepared_root / "prepared_data.npz",
            allow_pickle=False,
        ) as prepared_arrays:
            for split in ("validation", "test"):
                node_index = np.asarray(
                    prepared_arrays[f"{split}_node_index"],
                    dtype=np.int64,
                )
                canonical_views[split] = (
                    np.asarray(
                        prepared_arrays["target_expression"][node_index],
                        dtype=np.float32,
                    ),
                    np.asarray(
                        prepared_arrays["macroblock_ids"][node_index]
                    ),
                    np.arange(len(node_index), dtype=np.int64),
                )
                canonical_bundles[split] = load_fixed_mask_bundle(
                    prepared_root / "fixed_masks" / split
                )
    offset = (model_seed - 2) * 0.1
    evaluations: list[dict[str, object]] = []
    validation_metrics: list[dict[str, object]] = []
    test_metrics: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    splits = ("validation", "test") if sealed_test_opened else ("validation",)
    for split in splits:
        target, block_ids, cell_ids = canonical_views.get(
            split,
            (default_target, default_block_ids, default_cell_ids),
        )
        prediction = np.full_like(target, prediction_level + offset)
        if save_predictions:
            arrays[f"{split}__y_true"] = target
            arrays[f"{split}__block_ids"] = block_ids
            arrays[f"{split}__cell_ids"] = cell_ids
        for mode in modes:
            for replicate in replicates:
                prefix = f"{split}__{mode}__r{replicate}"
                if split in canonical_bundles:
                    mask = canonical_bundles[split].get(
                        split,
                        mode,
                        replicate,
                    )
                elif mode == "partial":
                    mask = np.zeros_like(target, dtype=bool)
                    mask[:, :2] = True
                else:
                    mask = np.ones_like(target, dtype=bool)
                metric_record = {
                    "split": split,
                    "mask_mode": mode,
                    "mask_replicate": replicate,
                    "mask_entry_id": f"{split}-{mode}-r{replicate}",
                    "metrics": {"huber": float(prediction_level)},
                }
                if split == "validation":
                    validation_metrics.append(metric_record)
                else:
                    test_metrics.append(metric_record)
                if not save_predictions:
                    continue
                prediction_key = f"{prefix}__prediction"
                mask_key = f"{prefix}__mask"
                arrays[prediction_key] = prediction
                arrays[mask_key] = mask
                evaluations.append(
                    {
                        "split": split,
                        "mask_mode": mode,
                        "mask_replicate": replicate,
                        "prefix": prefix,
                        "prediction_key": prediction_key,
                        "y_true_key": f"{split}__y_true",
                        "mask_key": mask_key,
                        "block_ids_key": f"{split}__block_ids",
                        "cell_ids_key": f"{split}__cell_ids",
                    }
                )
    (run_dir / "model_state.pt").write_bytes(b"synthetic checkpoint")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "validation": validation_metrics,
                "test": test_metrics,
                "test_targets_evaluated": sealed_test_opened,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if save_predictions:
        np.savez_compressed(run_dir / "predictions.npz", **arrays)
    job = dict(locked_job or {})
    graph_config = {
        "k": job.get("k", 12),
        "radius_um": job.get("radius_um", 50.0),
        "symmetry": job.get("symmetry", "union"),
        "min_distance_um": job.get("min_distance_um", 0.0),
    }
    model_config = {
        "name": model,
        "hidden_dim": job.get("hidden_dim", 256),
    }
    for field in ("graph_layers", "edge_embedding_dim"):
        if field in job:
            model_config[field] = job[field]
    training_config = {
        "model_seed": model_seed,
        "curriculum": job.get("curriculum", "P+N+B"),
        "max_epochs": job.get("max_epochs", 200),
        "patience": job.get("patience", 25),
        "amp": job.get("amp", True),
    }
    rewire = (
        {
            "seed": job.get("rewire_seed", 271828),
            "swaps_per_edge": job.get("swaps_per_edge", 1.0),
        }
        if graph_kind == "rewired"
        else None
    )
    manifest: dict[str, object] = {
        "format_version": 1,
        "artifact_kind": "normal_true_tissue_spatial_benchmark_run",
        "run_id": run_id,
        "status": "complete",
        "model_name": model,
        "model_seed": model_seed,
        "standards_lock": standards_lock,
        "prepared_artifact": prepared_artifact,
        "graph": {
            "graph_id": graph_id,
            "kind": graph_kind,
            "base_graph_id": base_graph_id,
            "config": graph_config,
            "edge_control": edge_control,
            "edge_control_seed": (
                job.get("rewire_seed")
                if edge_control == "permuted"
                else None
            ),
            "qc": _graph_qc(),
            "rewire": rewire,
        },
        "config": {
            "model": model_config,
            "run": {
                "model_seed": model_seed,
                "evaluate_test": sealed_test_opened,
                "save_predictions": save_predictions,
                "rewired": graph_kind == "rewired",
                "edge_control": edge_control,
                "edge_control_seed": (
                    job.get("rewire_seed")
                    if edge_control == "permuted"
                    else None
                ),
            },
            "graph": graph_config,
            "training": training_config,
        },
        "evaluations": evaluations,
        "metrics_file": "metrics.json",
        "artifacts": {
            "checkpoint": "model_state.pt",
            "predictions": "predictions.npz" if save_predictions else None,
        },
        "split_node_counts": {
            "train": 0 if prepared_artifact is not None else 16,
            "validation": len(
                canonical_views.get(
                    "validation",
                    (default_target, default_block_ids, default_cell_ids),
                )[0]
            ),
            "test": (
                len(
                    canonical_views.get(
                        "test",
                        (default_target, default_block_ids, default_cell_ids),
                    )[0]
                )
                if sealed_test_opened
                else None
            ),
        },
        "sealed_test_opened": sealed_test_opened,
    }
    manifest["files"] = {
        path.name: _sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    manifest["manifest_content_sha256"] = _manifest_content_sha256(manifest)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _write_complete_ladder(root: Path) -> None:
    for seed in range(5):
        _write_run(
            root,
            model="b0",
            model_seed=seed,
            graph_id="self",
            graph_kind="self",
            prediction_level=2.0,
        )
        _write_run(
            root,
            model="g1",
            model_seed=seed,
            graph_id="k12-r50-union",
            graph_kind="true",
            prediction_level=1.0,
        )
        _write_run(
            root,
            model="g1",
            model_seed=seed,
            graph_id="k12-r50-union__rewired",
            graph_kind="rewired",
            base_graph_id="k12-r50-union",
            prediction_level=1.5,
        )


def _write_recommendation(
    path: Path,
    selection: str,
    standard: dict[str, object],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "locked": True,
                "selection": selection,
                "candidate_id": f"{selection}-candidate",
                "required_seeds": 3,
                "standard": standard,
                "test_metrics_used": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _create_test_lock(root: Path) -> Path:
    recommendations = root / "recommendations"
    recommendations.mkdir(parents=True)
    paths = [
        _write_recommendation(
            recommendations / "graph.json",
            "graph",
            {
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
                "min_distance_um": 0.0,
            },
        ),
        _write_recommendation(
            recommendations / "mask.json",
            "mask",
            {"curriculum": "P+N+B"},
        ),
        _write_recommendation(
            recommendations / "hidden.json",
            "hidden",
            {"hidden_dim": 256},
        ),
        _write_recommendation(
            recommendations / "depth.json",
            "depth",
            {"graph_layers": 2},
        ),
        _write_recommendation(
            recommendations / "edge.json",
            "edge_embedding",
            {"edge_embedding_dim": 32},
        ),
    ]
    return create_standards_lock(paths, root / "standards-lock")


def _create_test_prepared(root: Path, *, replicates: int = 2) -> dict[str, object]:
    root.mkdir(parents=True)
    n_split_nodes = 16
    n_genes = 4
    grid = np.asarray(
        [(float(x), float(y)) for y in range(4) for x in range(4)],
        dtype=np.float64,
    )
    bundles: dict[str, dict[str, object]] = {}
    for split in ("validation", "test"):
        bundle = create_fixed_mask_bundle(
            {split: grid},
            n_genes,
            specs=(
                MaskSpec("partial", partial_gene_rate=0.5),
                MaskSpec("node", node_rate=0.5),
                MaskSpec("block", block_node_rate=0.5),
            ),
            replicates=replicates,
            base_seed=101,
        )
        directory = root / "fixed_masks" / split
        save_fixed_mask_bundle(bundle, directory)
        bundles[split] = {
            "bundle_id": bundle.bundle_id,
            "bundle_checksum": bundle.checksum,
            "directory": f"fixed_masks/{split}",
            "replicates": replicates,
        }

    target_expression = np.zeros(
        (2 * n_split_nodes, n_genes),
        dtype=np.float32,
    )
    macroblock_ids = np.concatenate(
        [
            np.repeat(np.arange(8, dtype=np.int64), 2),
            np.repeat(np.arange(8, dtype=np.int64), 2),
        ]
    )
    split_labels = np.asarray(
        ["val"] * n_split_nodes + ["test"] * n_split_nodes,
        dtype="U5",
    )
    arrays = {
        "target_expression": target_expression,
        "macroblock_ids": macroblock_ids,
        "split_labels": split_labels,
        "validation_node_index": np.arange(
            0,
            n_split_nodes,
            dtype=np.int64,
        ),
        "test_node_index": np.arange(
            n_split_nodes,
            2 * n_split_nodes,
            dtype=np.int64,
        ),
    }
    np.savez_compressed(
        root / "prepared_data.npz",
        **{name: arrays[name] for name in sorted(arrays)},
    )
    split_id = hashlib.sha256(b"split").hexdigest()[:16]
    manifest = {
        "format_version": 1,
        "artifact_kind": "normal_true_tissue_spatial_benchmark_preparation",
        "split": {"split_id": split_id},
        "fixed_masks": {"bundles": bundles},
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
            }
            for name, value in sorted(arrays.items())
        },
    }
    manifest["files"] = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.relative_to(root).as_posix()
        not in {"manifest.json", "checksums.sha256"}
    }
    content_checksum = _prepared_manifest_content_sha256(manifest)
    manifest["manifest_content_sha256"] = content_checksum
    manifest["artifact_id"] = content_checksum[:16]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_records = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.relative_to(root).as_posix() != "checksums.sha256"
    }
    (root / "checksums.sha256").write_text(
        "".join(
            f"{checksum}  {relative}\n"
            for relative, checksum in sorted(checksum_records.items())
        ),
        encoding="ascii",
    )
    return {
        "path": str(root.resolve()),
        "artifact_id": manifest["artifact_id"],
        "manifest_sha256": _sha256_file(manifest_path),
        "split_id": split_id,
        "validation_mask_bundle_id": bundles["validation"]["bundle_id"],
        "test_mask_bundle_id": bundles["test"]["bundle_id"],
    }


def _condition_graph(condition: str) -> tuple[str, str, str | None, str]:
    locked = "k12_r50_union_rbf8"
    if condition in {"b0", "b0_parameter_matched"}:
        return "self", "self", None, "none"
    if condition == "broad_field":
        return "broad-spatial-field", "broad_field", None, "none"
    if condition == "g1_rewired":
        return f"{locked}__rewired", "rewired", locked, "none"
    controls = {
        "g2_true": "none",
        "g2_zero": "zero",
        "g2_distance_only": "distance_only",
        "g2_permuted": "permuted",
    }
    return locked, "true", locked, controls.get(condition, "none")


def _write_locked_ladder(
    root: Path,
    standards_lock: Path,
    prepared: dict[str, object],
) -> None:
    _, lock, matrices = load_standards_lock(standards_lock)
    matrix = matrices[lock["final_execution"]["matrix_file"]]
    prediction_levels = {
        "b0": 2.0,
        "b0_parameter_matched": 1.9,
        "broad_field": 1.8,
        "b1": 1.4,
        "g1_true": 1.0,
        "g1_rewired": 1.5,
        "g2_true": 0.9,
        "g2_zero": 1.2,
        "g2_distance_only": 1.1,
        "g2_permuted": 1.15,
    }
    for job in expand_matrix(matrix):
        condition = condition_name(job)
        graph_id, graph_kind, base_graph_id, edge_control = _condition_graph(
            condition
        )
        _write_run(
            root,
            model=str(job["model"]),
            model_seed=int(job["seed"]),
            graph_id=graph_id,
            graph_kind=graph_kind,
            base_graph_id=base_graph_id,
            edge_control=edge_control,
            prediction_level=prediction_levels[condition],
            standards_lock=authorize_locked_test_job(
                standards_lock,
                job,
            ),
            prepared_artifact=prepared,
            locked_job=dict(job),
        )


@pytest.fixture
def experiment_output_ladder(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    _write_complete_ladder(runs)
    return runs


@pytest.fixture
def locked_output_ladder(tmp_path: Path) -> tuple[Path, Path]:
    standards_lock = _create_test_lock(tmp_path)
    prepared = _create_test_prepared(tmp_path / "prepared")
    runs = tmp_path / "locked-runs"
    _write_locked_ladder(runs, standards_lock, prepared)
    return runs, standards_lock


def test_multi_record_loader_and_five_seed_ensembles(
    experiment_output_ladder: Path,
) -> None:
    collection = load_run_collection(experiment_output_ladder)
    # 3 conditions x 5 seeds x 2 splits x 3 modes x 2 mask replicates.
    assert len(collection.records) == 180
    assert len(collection.manifest_checksums) == 15
    assert len(collection.prediction_checksums) == 15
    ensembles = ensemble_run_records(collection.records, expected_seeds=5)
    assert len(ensembles) == 36
    assert all(item.seed_complete for item in ensembles)
    assert all(item.seeds == (0, 1, 2, 3, 4) for item in ensembles)
    g1 = next(
        item
        for item in ensembles
        if item.model == "g1"
        and item.mask_mode == "node"
        and item.mask_replicate == 0
    )
    np.testing.assert_allclose(g1.ensemble_prediction, 1.0, atol=1e-7)


def test_locked_execution_preserves_all_conditions_and_rejects_contract_gaps(
    locked_output_ladder: tuple[Path, Path],
) -> None:
    runs, standards_lock = locked_output_ladder
    collection = load_run_collection(runs)
    execution = validate_locked_final_execution(
        collection,
        standards_lock,
        expected_seeds=5,
    )
    assert execution["complete"] is True
    assert execution["locked_seed_identities"] == [0, 1, 2, 3, 4]
    ensembles = ensemble_run_records(collection.records, expected_seeds=5)
    assert {
        item.condition for item in ensembles
        if item.split == "test" and item.mask_mode == "node"
    } == {
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
    }
    assert {
        item.condition for item in ensembles
        if item.model == "g2"
    } == {
        "g2_true",
        "g2_zero",
        "g2_distance_only",
        "g2_permuted",
    }
    assert all(
        record.graph_kind == "broad_field"
        for record in collection.records
        if record.condition == "broad_field"
    )

    removed_run = next(
        run_id for run_id, audit in collection.run_audits.items()
        if audit["condition"] == "b1" and audit["model_seed"] == 4
    )
    incomplete = replace(
        collection,
        records=tuple(
            record for record in collection.records
            if record.run_id != removed_run
        ),
        run_audits={
            run_id: audit
            for run_id, audit in collection.run_audits.items()
            if run_id != removed_run
        },
    )
    with pytest.raises(AnalysisContractError, match="execution is incomplete"):
        validate_locked_final_execution(incomplete, standards_lock)

    target_run = next(iter(collection.run_audits))
    incomplete_masks = copy.deepcopy(dict(collection.run_audits))
    audit = dict(incomplete_masks[target_run])
    audit["evaluation_keys"] = tuple(audit["evaluation_keys"][:-1])
    incomplete_masks[target_run] = audit
    with pytest.raises(AnalysisContractError, match="exact fixed-mask"):
        validate_locked_final_execution(
            replace(collection, run_audits=incomplete_masks),
            standards_lock,
        )

    mismatched_config = copy.deepcopy(dict(collection.run_audits))
    config_audit = dict(mismatched_config[target_run])
    config = copy.deepcopy(config_audit["config"])
    config["model"]["hidden_dim"] = 999
    config_audit["config"] = config
    mismatched_config[target_run] = config_audit
    with pytest.raises(AnalysisContractError, match="differs from its locked job"):
        validate_locked_final_execution(
            replace(collection, run_audits=mismatched_config),
            standards_lock,
        )

    mismatched_lock = copy.deepcopy(dict(collection.run_audits))
    lock_audit = dict(mismatched_lock[target_run])
    authorization = dict(lock_audit["standards_lock"])
    authorization["final_matrix_sha256"] = "0" * 64
    lock_audit["standards_lock"] = authorization
    mismatched_lock[target_run] = lock_audit
    with pytest.raises(AnalysisContractError, match="mismatched lock field"):
        validate_locked_final_execution(
            replace(collection, run_audits=mismatched_lock),
            standards_lock,
        )


def test_locked_execution_rejects_consistently_altered_fixed_masks(
    locked_output_ladder: tuple[Path, Path],
) -> None:
    """Canonical masks, not merely mutually paired masks, are required."""

    runs, standards_lock = locked_output_ladder
    collection = load_run_collection(runs)
    altered_records = []
    for record in collection.records:
        if record.evaluation_key == ("test", "node", 0):
            altered_mask = np.array(record.mask, copy=True)
            altered_mask[0, 0] = ~altered_mask[0, 0]
            altered_records.append(replace(record, mask=altered_mask))
        else:
            altered_records.append(record)
    altered = replace(collection, records=tuple(altered_records))

    # Every condition and every seed has the same altered mask, so the older
    # pairwise-only contract would have accepted this collection.
    target_records = [
        record
        for record in altered.records
        if record.evaluation_key == ("test", "node", 0)
    ]
    assert len(target_records) == 50
    assert all(
        np.array_equal(target_records[0].mask, record.mask)
        for record in target_records[1:]
    )
    with pytest.raises(
        AnalysisContractError,
        match="exact canonical prepared fixed mask",
    ):
        validate_locked_final_execution(altered, standards_lock)


def test_locked_execution_rejects_noncanonical_prepared_rows(
    locked_output_ladder: tuple[Path, Path],
) -> None:
    runs, standards_lock = locked_output_ladder
    collection = load_run_collection(runs)
    mutations = (
        ("y_true", "canonical prepared target rows"),
        ("block_ids", "canonical prepared split rows"),
        ("cell_ids", "canonical prepared split-local row order"),
    )
    for field, message in mutations:
        altered_records = []
        for record in collection.records:
            if record.split != "test":
                altered_records.append(record)
                continue
            value = np.array(getattr(record, field), copy=True)
            if field == "y_true":
                value[0, 0] += 1.0
            else:
                value[0] = value[0] + 1
            altered_records.append(replace(record, **{field: value}))
        with pytest.raises(AnalysisContractError, match=message):
            validate_locked_final_execution(
                replace(collection, records=tuple(altered_records)),
                standards_lock,
            )


def test_locked_execution_rejects_tampered_prepared_mask_file(
    locked_output_ladder: tuple[Path, Path],
) -> None:
    runs, standards_lock = locked_output_ladder
    collection = load_run_collection(runs)
    prepared = next(iter(collection.run_audits.values()))["prepared_artifact"]
    masks_path = (
        Path(str(prepared["path"]))
        / "fixed_masks"
        / "test"
        / "masks.npz"
    )
    with masks_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(
        AnalysisContractError,
        match="Prepared artifact could not be verified",
    ):
        validate_locked_final_execution(collection, standards_lock)


def test_analysis_applies_locked_gate_and_writes_auditable_outputs(
    tmp_path: Path,
    locked_output_ladder: tuple[Path, Path],
) -> None:
    pytest.importorskip("matplotlib")
    runs, standards_lock = locked_output_ladder
    output = tmp_path / "summary"
    summary = analyze_run_directory(
        runs,
        output,
        standards_lock_dir=standards_lock,
        locked_graph_id="k12_r50_union_rbf8",
        expected_seeds=5,
        n_bootstrap=300,
        bootstrap_seed=19,
    )
    gate = summary["acceptance_gate"]
    assert summary["graph_candidate_split"] == "validation"
    assert gate["outcome"] == "supported_within_core"
    assert gate["supported"] is True
    assert gate["conclusion_split_complete"] is True
    assert gate["required_conclusion_split"] == "test"
    assert gate["five_seed_complete"] is True
    assert gate["paired_mask_replicates_complete"] is True
    assert all(item["passed"] for item in gate["criteria"])
    assert summary["execution_completeness"]["complete"] is True
    assert summary["execution_completeness"]["observed_run_count"] == 50
    assert len(summary["secondary_paired_gains"]) == 21
    assert {
        row["condition"] for row in summary["loss_summary"]
    } == {
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
    }
    assert "does not establish patient-level generalisation" in summary["scope"]
    assert "biological effect sizes" in summary["interpretation"]

    primary = next(
        gain
        for gain in summary["paired_spatial_gains"]
        if gain["comparison"] == "B0_minus_G1"
        and gain["mask_mode"] == "node"
    )
    assert primary["n_spatial_blocks"] == 8
    assert primary["n_mask_replicates"] == 2
    assert primary["baseline_seed_counts"] == [5]
    assert primary["spatial_seed_counts"] == [5]
    assert primary["relative_gain"] > 0.02
    assert primary["delta_ci_lower"] > 0

    expected_files = [
        "summary.json",
        "model_mode_losses.csv",
        "spatial_gains.csv",
        "acceptance_gate.csv",
        "graph_qc.csv",
        "condition_coverage.csv",
        "figures/loss_by_model_mode.png",
        "figures/graph_candidate_loss.png",
        "figures/spatial_gain_ci.png",
        "figures/graph_qc_degree_distance.png",
    ]
    for relative in expected_files:
        path = output / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 100
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert path.stat().st_size > 10_000

    saved = json.loads((output / "summary.json").read_text())
    assert saved["acceptance_gate"] == gate
    assert len(saved["artifact_checksums"]) == 9
    assert all(
        len(checksum) == 64
        for checksum in saved["artifact_checksums"].values()
    )
    assert not any(
        isinstance(value, float) and not np.isfinite(value)
        for value in saved.values()
    )
    with (output / "acceptance_gate.csv").open(newline="", encoding="utf-8") as handle:
        gate_rows = list(csv.DictReader(handle))
    assert {row["category"] for row in gate_rows} == {
        "input_validity",
        "hypothesis_gate",
    }


def test_unpaired_fixed_mask_is_rejected_before_seed_averaging() -> None:
    target = np.zeros((4, 3), dtype=np.float32)
    prediction = np.ones_like(target)
    first_mask = np.ones_like(target, dtype=bool)
    second_mask = first_mask.copy()
    second_mask[0, 0] = False
    common = {
        "condition": "b0",
        "model": "b0",
        "graph_id": "self",
        "graph_kind": "self",
        "base_graph_id": None,
        "graph_config": {},
        "edge_control": "none",
        "split": "test",
        "mask_mode": "node",
        "mask_replicate": 0,
        "y_true": target,
        "prediction": prediction,
        "block_ids": np.arange(4),
        "cell_ids": np.asarray(["a", "b", "c", "d"]),
        "manifest_path": "manifest.json",
        "manifest_checksum": "0" * 64,
    }
    first = PredictionRecord(
        run_id="run-0",
        model_seed=0,
        mask=first_mask,
        **common,
    )
    second = PredictionRecord(
        run_id="run-1",
        model_seed=1,
        mask=second_mask,
        **common,
    )
    with pytest.raises(AnalysisContractError, match="fixed masks/row order differ"):
        ensemble_run_records([first, second], expected_seeds=2)


def test_incomplete_or_failed_prespecified_gate_never_claims_support() -> None:
    primary = {
        "comparison": "B0_minus_G1",
        "mask_mode": "node",
        "split": "test",
        "seed_complete": True,
        "replicate_pair_complete": True,
        "baseline_seed_counts": [5],
        "spatial_seed_counts": [5],
        "relative_gain": 0.01,
        "delta_ci_lower": -0.01,
        "delta": 0.1,
    }
    block = {
        **primary,
        "mask_mode": "block",
        "relative_gain": 0.03,
        "delta": -0.02,
    }
    rewired = {
        **primary,
        "comparison": "rewired_G1_minus_true_G1",
        "delta": -0.04,
    }
    rewired_block = {**rewired, "mask_mode": "block"}
    gate = evaluate_acceptance_gate(
        [primary, block, rewired, rewired_block]
    )
    assert gate["outcome"] == "not_supported_by_prespecified_gate"
    assert gate["supported"] is False
    assert not any(item["passed"] for item in gate["criteria"])
    without_rewired_block = evaluate_acceptance_gate(
        [primary, block, rewired]
    )
    assert without_rewired_block["input_complete"] is True
    assert without_rewired_block["outcome"] == (
        "not_supported_by_prespecified_gate"
    )

    incomplete = evaluate_acceptance_gate([primary])
    assert incomplete["outcome"] == "incomplete"
    assert incomplete["supported"] is False

    validation_only = [
        {**gain, "split": "validation"}
        for gain in (primary, block, rewired, rewired_block)
    ]
    validation_gate = evaluate_acceptance_gate(validation_only)
    assert validation_gate["outcome"] == "incomplete"
    assert validation_gate["supported"] is False
    assert validation_gate["conclusion_split_complete"] is False
    assert not any(item["passed"] for item in validation_gate["criteria"])


def test_loader_matches_experiment_manifest_and_shared_npz_schema(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path / "runs",
        model="g1",
        model_seed=2,
        graph_id="k12_r50_union_rbf8",
        graph_kind="true",
        prediction_level=1.0,
    )
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == (
        "normal_true_tissue_spatial_benchmark_run"
    )
    assert manifest["sealed_test_opened"] is True
    declaration = manifest["evaluations"][0]
    assert set(declaration) == {
        "split",
        "mask_mode",
        "mask_replicate",
        "prefix",
        "prediction_key",
        "y_true_key",
        "mask_key",
        "block_ids_key",
        "cell_ids_key",
    }
    with np.load(run / "predictions.npz", allow_pickle=False) as archive:
        assert declaration["prediction_key"] in archive.files
        assert declaration["y_true_key"] == "validation__y_true"
        assert "validation__block_ids" in archive.files
        assert "test__y_true" in archive.files
        assert "test__node__r0__prediction" in archive.files

    collection = load_run_collection(run)
    assert len(collection.records) == 12
    assert {record.split for record in collection.records} == {
        "validation",
        "test",
    }
    assert collection.excluded_runs == ()
    assert collection.prediction_checksums == {
        f"{manifest['run_id']}/predictions.npz": manifest["files"][
            "predictions.npz"
        ]
    }

    with (run / "predictions.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(AnalysisContractError, match="Prediction checksum mismatch"):
        load_run_collection(run)

    content_run = _write_run(
        tmp_path / "content-tamper",
        model="b0",
        model_seed=0,
        graph_id="self",
        graph_kind="self",
        prediction_level=2.0,
    )
    content_manifest_path = content_run / "manifest.json"
    content_manifest = json.loads(content_manifest_path.read_text())
    content_manifest["config"]["model"]["hidden_dim"] = 999
    content_manifest_path.write_text(
        json.dumps(content_manifest),
        encoding="utf-8",
    )
    with pytest.raises(
        AnalysisContractError,
        match="manifest content checksum mismatch",
    ):
        load_run_collection(content_run)


def test_screening_and_incomplete_runs_are_excluded_from_conclusions(
    tmp_path: Path,
    locked_output_ladder: tuple[Path, Path],
) -> None:
    pytest.importorskip("matplotlib")
    runs, standards_lock = locked_output_ladder
    screening = _write_run(
        runs,
        model="g1",
        model_seed=7,
        graph_id="k8_r30_mutual_rbf8",
        graph_kind="true",
        prediction_level=0.01,
        sealed_test_opened=False,
        save_predictions=True,
    )
    screening_id = json.loads(
        (screening / "manifest.json").read_text()
    )["run_id"]
    no_prediction_screening = _write_run(
        runs,
        model="b1",
        model_seed=8,
        graph_id="k16_r75_union_rbf8",
        graph_kind="true",
        prediction_level=0.01,
        sealed_test_opened=False,
        save_predictions=False,
    )
    no_prediction_screening_id = json.loads(
        (no_prediction_screening / "manifest.json").read_text()
    )["run_id"]
    incomplete = runs / "unfinished"
    incomplete.mkdir()
    incomplete_id = "f" * 20
    (incomplete / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "artifact_kind": (
                    "normal_true_tissue_spatial_benchmark_run"
                ),
                "run_id": incomplete_id,
                "status": "incomplete",
                "model_name": "g1",
                "model_seed": 0,
            }
        ),
        encoding="utf-8",
    )

    collection = load_run_collection(runs)
    assert len(collection.records) == 600
    excluded = {item["run_id"]: item for item in collection.excluded_runs}
    assert excluded[screening_id]["reason"] == "sealed_test_not_opened"
    assert excluded[screening_id]["phase"] == "screening_or_validation"
    assert (
        excluded[no_prediction_screening_id]["reason"]
        == "sealed_test_not_opened"
    )
    assert excluded[incomplete_id]["reason"] == "run_not_complete"
    assert all(
        record.run_id
        not in {
            screening_id,
            no_prediction_screening_id,
            incomplete_id,
        }
        for record in collection.records
    )
    with pytest.raises(
        AnalysisContractError,
        match="No conclusion-eligible sealed-test",
    ):
        load_run_collection(screening)

    output = tmp_path / "summary"
    summary = analyze_run_directory(
        runs,
        output,
        standards_lock_dir=standards_lock,
        locked_graph_id="k12_r50_union_rbf8",
        expected_seeds=5,
        n_bootstrap=100,
    )
    assert summary["acceptance_gate"]["supported"] is True
    assert summary["conclusion_input_policy"] == {
        "required_run_status": "complete",
        "required_sealed_test_opened": True,
        "required_evaluation_split": "test",
        "screening_and_incomplete_runs_excluded": True,
        "required_verified_standards_lock": True,
        "required_exact_final_matrix": True,
    }
    assert {item["run_id"] for item in summary["excluded_nonconclusion_runs"]} == {
        screening_id,
        no_prediction_screening_id,
        incomplete_id,
    }
    assert screening_id not in summary["input_run_ids"]
    assert no_prediction_screening_id not in summary["input_run_ids"]
    assert incomplete_id not in summary["input_run_ids"]
    assert screening_id not in summary["input_manifest_checksums"]
    assert screening_id in summary["discovered_manifest_checksums"]
    with pytest.raises(
        AnalysisContractError,
        match="screening/validation splits are not conclusion-bearing",
    ):
        analyze_run_directory(
            runs,
            tmp_path / "invalid-validation-conclusion",
            standards_lock_dir=standards_lock,
            locked_graph_id="k12_r50_union_rbf8",
            split="validation",
            n_bootstrap=100,
        )


def test_single_record_npz_aliases_and_cli_help(tmp_path: Path) -> None:
    run = tmp_path / "single"
    run.mkdir()
    target = np.zeros((4, 2), dtype=np.float32)
    np.savez_compressed(
        run / "predictions.npz",
        y_pred=np.ones_like(target),
        target_expression=target,
        gene_mask=np.ones_like(target, dtype=np.uint8),
        macroblock_ids=np.arange(4),
        mask_mode=np.asarray("whole_node"),
        split=np.asarray("test"),
        mask_replicate=np.asarray(0),
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "single",
                "status": "complete",
                "model": {"name": "b0"},
                "model_seed": 0,
                "graph_kind": "self",
                "prediction_file": "predictions.npz",
                "sealed_test_opened": True,
            }
        ),
        encoding="utf-8",
    )
    collection = load_run_collection(run)
    assert len(collection.records) == 1
    assert collection.records[0].mask_mode == "node"
    assert collection.records[0].mask.dtype == bool
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("sealed_test_opened")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        AnalysisContractError,
        match="boolean sealed-test declaration",
    ):
        load_run_collection(run)
    manifest["sealed_test_opened"] = True
    manifest["prediction_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AnalysisContractError, match="Prediction checksum mismatch"):
        load_run_collection(run)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analysis" / "analyze_spatial_benchmark.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--locked-graph-id" in result.stdout
    assert "--standards-lock" in result.stdout
    assert "within-core" in result.stdout

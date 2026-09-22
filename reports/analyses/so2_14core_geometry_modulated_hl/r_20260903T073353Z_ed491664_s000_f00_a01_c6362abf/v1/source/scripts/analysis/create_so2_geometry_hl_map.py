"""Versioned, registered hL PNG report for the completed geometry-QKV run."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gc
import inspect
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch

from spatial_benchmark.checkpoint_catalog import resolve_checkpoint
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.geometry_modulated_relative_qkv_graph_transformer import (
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer as Model,
)
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.relative_qkv_embedding_clustering import (
    _array_sha256, _atomic_write_json, _atomic_write_text, _file_manifest,
    _file_record, _git_provenance, _project_artifact_path, _read_json, _read_yaml,
    _receipt_with_self_hash, _tensor_sha256, _verify_self_hash,
    _write_deterministic_npz,
)
from spatial_benchmark.relative_qkv_post_training import _tree_sha256
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle
from spatial_benchmark.so2_hl_clustering import (
    SO2ResolvedInputs, cluster_joint_contextual_embeddings, load_contextual_core,
)
from spatial_benchmark.so2_pooled_full_core import EXPECTED_TOTAL_CELLS, SO2_CORE_NUMBERS
from spatial_benchmark.so2_recurrent_hl_clustering import (
    _balanced_assignments, _device_inventory, _load_core_batch, _source_core_records,
    extract_full_recurrent_contextual_embedding as extract_hl,
    verify_leiden_determinism,
)
from spatial_benchmark.so2_relative_graphs import (
    _verify_so2_cohort_manifest, _verify_so2_graph_collection,
)

RUN_ID = "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf"
CAMPAIGN = "cmp_20260903_so2_14core_geometry_modulated_relative_qkv_seed0_batch2"
REPORT = "so2_14core_geometry_modulated_hl"
EVALUATION = "ev_so2_geometry_modulated_hl_ed491664_v1"
SEED = 20260825


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def event(root, name, **values):
    record = {"time": now(), "event": name, **values}
    with (root / "events.jsonl").open("a") as out:
        out.write(json.dumps(record, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(record, sort_keys=True), flush=True)


def load_model(checkpoint, expected_sha):
    require(sha256_file(checkpoint) == expected_sha, "Checkpoint checksum changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    construction = payload["model_construction"]
    require(payload["run_id"] == RUN_ID and payload["campaign_id"] == CAMPAIGN,
            "Wrong checkpoint identity")
    require(construction["class"] == Model.__name__, "Wrong model architecture")
    require(payload["completed_global_epochs"] == 200, "Wrong checkpoint epoch")
    parameters = inspect.signature(Model).parameters
    require(set(parameters).issubset(construction), "Missing constructor arguments")
    model = Model(**{key: construction[key] for key in parameters})
    state = payload["model_state_dict"]
    require(_tree_sha256(state) == payload["model_state_checksum"], "Bad state hash")
    model.load_state_dict(state, strict=True)
    require(_tree_sha256(model.state_dict()) == payload["model_state_checksum"],
            "Reload changed model weights")
    require(sum(p.numel() for p in model.parameters()) == 5134088, "Parameter drift")
    model.eval()
    require(not any(m.training for m in model.modules()), "Training mode is enabled")
    return model


def verify_with_training_archive_code(bundle, stage):
    """Use the provenance-bound validator that recognizes this training protocol."""
    git = _read_json(bundle / "provenance/git.json", label="training Git")
    relative = "src/spatial_benchmark/run_archive.py"
    patch = (bundle / "provenance/uncommitted_changes.patch").read_text()
    header = f"diff --git a/{relative} b/{relative}\n"
    section = header + patch.split(header, 1)[1].split("\ndiff --git ", 1)[0] + "\n"
    destination = stage / "training_verifier" / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(subprocess.check_output(["git", "show", f"{git['commit']}:{relative}"]))
        subprocess.run(["git", "apply", "--unsafe-paths", "--directory", str(stage / "training_verifier")],
                       input=section, text=True, check=True, capture_output=True)
    expected_blob = section.split("index ", 1)[1].split("..", 1)[1].split()[0]
    actual_blob = subprocess.check_output(["git", "hash-object", str(destination)], text=True).strip()
    require(actual_blob.startswith(expected_blob), "Recovered archive validator differs from training patch")
    name = "spatial_benchmark._geometry_hl_training_archive"
    spec = importlib.util.spec_from_file_location(name, destination)
    module = importlib.util.module_from_spec(spec)
    os.sys.modules[name] = module
    spec.loader.exec_module(module)
    verification = module.verify_run_bundle(bundle)
    _atomic_write_json(stage / "source_bundle_verification.json", {
        "verification": verification, "validator_git_blob": actual_blob,
        "validator_file": _file_record(destination), "training_patch": _file_record(bundle / "provenance/uncommitted_changes.patch")})


def resolve_inputs(paths, registry, stage):
    run = registry.show_run(RUN_ID)
    require(run is not None and run["status"] == "completed", "Source run not completed")
    checkpoint = resolve_checkpoint(registry, RUN_ID, paths, role="last")
    bundle = RunArchive.artifact_path_for(RUN_ID, paths)
    verify_with_training_archive_code(bundle, stage)
    config = _read_yaml(bundle / "config.resolved.yaml", label="training config")
    dataset = config["dataset"]
    cohort = _project_artifact_path(dataset["prepared_artifact"], paths=paths)
    graph = _project_artifact_path(dataset["prepared_graph_artifact"], paths=paths)
    for root, key in ((cohort, "cohort_manifest_file_sha256"),
                      (graph, "graph_manifest_file_sha256")):
        require(sha256_file(root / "manifest.json") == dataset[key], "Source manifest drift")
    cohort_manifest = _verify_so2_cohort_manifest(cohort)
    graph_manifest = _verify_so2_graph_collection(graph, cohort_manifest_path=cohort / "manifest.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = {
        "selection_policy": "latest_completed_training_run_by_end_time_as_of_2026_09_05",
        "run_id": RUN_ID, "campaign_id": CAMPAIGN, "epoch": 200, "model_seed": 0,
        "checkpoint": {"path": str(checkpoint), **_file_record(checkpoint)},
        "model_state_sha256": payload["model_state_checksum"],
        "source_bundle_manifest": _file_record(bundle / "manifest.yaml"),
        "cohort_manifest": _file_record(cohort / "manifest.json"),
        "graph_manifest": _file_record(graph / "manifest.json"),
        "training_source": _read_json(bundle / "provenance/git.json", label="training Git"),
        "analysis_source": _git_provenance(paths.project_root),
        "dataset_id": dataset["dataset_id"], "split_id": dataset["split_id"],
        "split_fingerprint": dataset["split_fingerprint"],
        "fit_only": True, "independent_patient_replication": False,
    }
    return SO2ResolvedInputs(RUN_ID, paths.project_root, checkpoint, sha256_file(checkpoint),
                             payload, bundle, cohort, graph, cohort_manifest, graph_manifest, provenance)


def verify_core(root, record, checkpoint_sha):
    receipt = _read_json(root / "embeddings" / f"core_{record['core_number']}_receipt.json", label="core receipt")
    _verify_self_hash(receipt, label="core receipt")
    require(receipt["run_id"] == RUN_ID and receipt["checkpoint_sha256"] == checkpoint_sha,
            "Embedding checkpoint identity changed")
    require(receipt["source_core"] == record, "Core source record changed")
    path = root / receipt["embedding_file"]
    require(_file_record(path) == receipt["file"], "Embedding checksum changed")
    core = load_contextual_core(path, alias=record["alias"], core_number=record["core_number"],
                                expected_cells=record["cell_count"])
    require(core.hL.shape[1] == 256, "Embedding width drift")
    require(_array_sha256("hL", core.hL) == receipt["hL_array_sha256"], "Embedding array hash drift")
    return receipt


def extract_assignment(job):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(job["device"])
    torch.cuda.set_device(device)
    model = load_model(Path(job["checkpoint"]), job["checkpoint_sha256"]).to(device)
    root = Path(job["root"])
    records = []
    for record in job["records"]:
        number = record["core_number"]
        receipt_path = root / "embeddings" / f"core_{number}_receipt.json"
        if receipt_path.exists():
            records.append(verify_core(root, record, job["checkpoint_sha256"]))
            continue
        batch, coordinates = _load_core_batch(cohort_dir=Path(job["cohort"]),
            graph_dir=Path(job["graph"]), record=record)
        expression_hash = _tensor_sha256("target_expression", batch.target_expression)
        metadata_hash = _tensor_sha256("node_covariates", batch.node_covariates)
        expression = batch.target_expression.to(device)
        covariates = batch.node_covariates.to(device)
        mask = torch.zeros_like(expression, dtype=torch.bool)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.monotonic()
        with torch.inference_mode():
            tensor = extract_hl(model, expression, mask, batch.edge_index, batch.relative_geometry, covariates)
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        hl = tensor.cpu().numpy().copy()
        repeat = None
        if number == 21:
            with torch.inference_mode():
                repeated = extract_hl(model, expression, mask, batch.edge_index, batch.relative_geometry, covariates).cpu().numpy()
            repeat = {"allclose": bool(np.allclose(hl, repeated, rtol=1e-6, atol=1e-5)),
                "exact_match": bool(np.array_equal(hl, repeated)),
                "max_absolute_difference": float(np.max(np.abs(hl - repeated))),
                "rtol": 1e-6, "atol": 1e-5}
            require(repeat["allclose"], "Pilot extraction not repeatable")
            del repeated
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        free, total = torch.cuda.mem_get_info(device)
        require(total - reserved >= 2 * 1024**3 and free >= 2 * 1024**3, "Insufficient GPU margin")
        require(expression_hash == _tensor_sha256("target_expression", batch.target_expression), "Input mutation")
        require(metadata_hash == _tensor_sha256("node_covariates", batch.node_covariates), "Metadata mutation")
        path = root / "embeddings" / f"core_{number}_hL.npz"
        require(not path.exists(), "Unreceipted embedding exists; inspect before retry")
        _write_deterministic_npz(path, {"cell_index": np.arange(record["cell_count"], dtype=np.int64),
            "core_number": np.asarray(number, dtype=np.int16), "coordinates_um": coordinates, "hL": hl})
        receipt = _receipt_with_self_hash({"schema": "so2_geometry_hl_core_v1", "status": "complete",
            "run_id": RUN_ID, "checkpoint_sha256": job["checkpoint_sha256"], "created_at": now(),
            "alias": record["alias"], "core_number": number, "cell_count": record["cell_count"],
            "source_core": record, "embedding_file": str(path.relative_to(root)), "file": _file_record(path),
            "hL_shape": list(hl.shape), "hL_array_sha256": _array_sha256("hL", hl),
            "source_expression_sha256": expression_hash, "source_metadata_sha256": metadata_hash,
            "repeat": repeat, "inference": {"device": str(device), "dtype": "float32", "tf32": False,
                "model_eval": True, "torch_inference_mode": True, "gene_mask_nonzero_count": 0,
                "complete_core": True, "graph_device": "cpu", "geometry_device": "cpu",
                "elapsed_seconds": elapsed, "cells_per_second": record["cell_count"] / elapsed,
                "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}})
        _atomic_write_json(receipt_path, receipt)
        records.append(verify_core(root, record, job["checkpoint_sha256"]))
        print(json.dumps({"core": number, "device": str(device), "seconds": round(elapsed, 2),
                          "peak_vram_gib": round(allocated / 1024**3, 3)}), flush=True)
        del batch, coordinates, expression, covariates, mask, tensor, hl
        model.clear_edge_layout_cache()
        gc.collect()
        torch.cuda.empty_cache()
    return records


def run_jobs(jobs):
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        return [record for result in pool.map(extract_assignment, jobs) for record in result]


def verify_report(root):
    manifest = _read_json(root / "manifest.json", label="report manifest")
    _verify_self_hash(manifest, label="report manifest")
    require(manifest["run_id"] == RUN_ID and manifest["status"] == "complete", "Wrong report")
    require((root / "_SUCCESS").read_text().strip() == sha256_file(root / "manifest.json"), "Bad completion marker")
    for relative, expected in manifest["files"].items():
        require(_file_record(root / relative) == expected, f"Report checksum failed: {relative}")
    extraction = _read_json(root / "embeddings/extraction_manifest.json", label="extraction")
    require([r["core_number"] for r in extraction["cores"]] == list(SO2_CORE_NUMBERS), "Core order drift")
    for receipt in extraction["cores"]:
        verify_core(root, receipt["source_core"], manifest["provenance"]["checkpoint"]["sha256"])
    from PIL import Image
    with Image.open(root / manifest["png"]) as image:
        image.verify()
    return manifest


def register_report(registry, root, manifest):
    with registry.transaction(immediate=True) as connection:
        connection.execute("UPDATE evaluations SET status=?, metrics_json=?, artifact_path=? WHERE evaluation_id=? AND run_id=?",
            ("completed", json.dumps({"total_cells": EXPECTED_TOTAL_CELLS, "cluster_count": manifest["cluster_count"],
                                      "descriptive_only": True}), str(root), EVALUATION, RUN_ID))
        for relative in ("manifest.json", manifest["png"]):
            path = root / relative
            if connection.execute("SELECT 1 FROM artifacts WHERE evaluation_id=? AND path=?", (EVALUATION, str(path))).fetchone():
                continue
            connection.execute("INSERT INTO artifacts(run_id,evaluation_id,kind,path,sha256,size_bytes,status,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (RUN_ID, EVALUATION, "posthoc_hl_png_v1" if path.suffix == ".png" else "posthoc_hl_manifest_v1",
                 str(path), sha256_file(path), path.stat().st_size, "present", now()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    paths = current_paths()
    registry = Registry(initialize=False)
    final = paths.report_root / "analyses" / REPORT / RUN_ID / "v1"
    stage = paths.scratch_root / "active_runs" / RUN_ID / "posthoc_reports" / REPORT / "v1"
    if final.exists() or args.verify_only:
        manifest = verify_report(final)
        if not args.verify_only:
            register_report(registry, final, manifest)
        print(json.dumps({"status": "verified", "png": str(final / manifest["png"])}))
        return
    stage.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        event(stage, "verify_source_inputs")
        inputs = resolve_inputs(paths, registry, stage)
        records = _source_core_records(inputs)
        devices = [f"cuda:{i}" for i in range(4)]
        inventory = _device_inventory(devices)
        smi = subprocess.check_output(["nvidia-smi"], text=True)
        _atomic_write_text(stage / "nvidia-smi.txt", smi)
        _atomic_write_text(stage / "nvidia-topology.txt", subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True))
        parameters = {"schema": "so2_geometry_hl_report_v1", "run_id": RUN_ID, "evaluation_id": EVALUATION,
            "classification": {"lifecycle_stage": "exploratory_screen", "study_axis": "hL_visualization",
                               "scientific_variant": "unmasked_hl_pca50_cosine30_leiden1", "seed": SEED, "version": 1},
            "model_seed": 0, "epoch": 200, "gene_mask": "all_zero_fully_observed", "pca_components": 50,
            "n_neighbors": 30, "leiden_resolution": 1.0, "seed": SEED, "dpi": 300,
            "devices": devices, "dtype": "float32", "cpu_threads_per_worker": 4}
        config_path = stage / "config.resolved.json"
        if config_path.exists():
            require(_read_json(config_path, label="staged config") == parameters, "Staged parameters changed")
        else:
            _atomic_write_json(config_path, parameters)
        with registry.connect() as connection:
            existing = connection.execute("SELECT * FROM evaluations WHERE evaluation_id=?", (EVALUATION,)).fetchone()
        if existing is None:
            registry.create_evaluation(EVALUATION, run_id=RUN_ID, checkpoint_name="last",
                dataset_id=inputs.provenance["dataset_id"], split_id=inputs.provenance["split_id"],
                status="running", metrics={"analysis_type": "descriptive_hL_spatial_map", "config": parameters}, artifact_path=stage)
        else:
            require(existing["run_id"] == RUN_ID, "Evaluation ID collision")
        _atomic_write_json(stage / "provenance.json", dict(inputs.provenance))
        _atomic_write_json(stage / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "devices": inventory,
            "command": "PYTHONPATH=src /venv/main/bin/python scripts/analysis/create_so2_geometry_hl_map.py", "cwd": str(paths.project_root)})
        _atomic_write_text(stage / "dependencies.txt", subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True))
        sources = [Path(__file__), *[paths.project_root / "src/spatial_benchmark" / name for name in (
            "geometry_modulated_relative_qkv_graph_transformer.py", "relative_qkv_graph_transformer.py",
            "so2_geometry_hl_figures.py", "so2_hl_clustering.py", "so2_recurrent_hl_clustering.py",
            "relative_qkv_embedding_clustering.py", "so2_relative_graphs.py")]]
        for source in sources:
            destination = stage / "source" / source.relative_to(paths.project_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        contract = paths.project_root / "experiments/campaigns" / CAMPAIGN / "HL_CLUSTERING_ANALYSIS.md"
        shutil.copyfile(contract, stage / "TASK_CONTRACT.md")
        base = {"checkpoint": str(inputs.checkpoint_path), "checkpoint_sha256": inputs.checkpoint_sha256,
                "cohort": str(inputs.cohort_dir), "graph": str(inputs.graph_dir), "root": str(stage)}
        pilots = [r for r in records if r["core_number"] in (21, 23)]
        event(stage, "pilot_start", cores=[21, 23], devices=devices[:2])
        run_jobs([{**base, "device": device, "records": [record]} for device, record in zip(devices, pilots)])
        for record in pilots:
            pilot = verify_core(stage, record, inputs.checkpoint_sha256)
            if record["core_number"] == 21:
                require(pilot["repeat"]["allclose"], "Pilot repeat failed")
        event(stage, "pilot_passed")
        remaining = [r for r in records if r["core_number"] not in (21, 23)]
        schedule = _balanced_assignments(remaining, devices)
        _atomic_write_json(stage / "schedule.json", {device: [r["core_number"] for r in rows] for device, rows in schedule.items()})
        event(stage, "full_extraction_start", schedule={device: [r["core_number"] for r in rows] for device, rows in schedule.items()})
        run_jobs([{**base, "device": device, "records": rows} for device, rows in schedule.items() if rows])
        receipts = [verify_core(stage, record, inputs.checkpoint_sha256) for record in records]
        extraction = _receipt_with_self_hash({"schema": "so2_geometry_hl_extraction_v1", "status": "complete", "run_id": RUN_ID,
            "checkpoint_sha256": inputs.checkpoint_sha256, "total_cells": EXPECTED_TOTAL_CELLS, "cores": receipts})
        _atomic_write_json(stage / "embeddings/extraction_manifest.json", extraction)
        event(stage, "clustering_start", cells=EXPECTED_TOTAL_CELLS)
        torch.set_num_threads(8)
        clustering = cluster_joint_contextual_embeddings(inputs=inputs, output_root=stage, extraction_receipt=extraction,
            n_neighbors=30, pca_components=50, leiden_resolution=1.0, random_seed=SEED)
        event(stage, "leiden_replay_start", clusters=clustering["cluster_count"])
        determinism = verify_leiden_determinism(output_root=stage, clustering_receipt=clustering, resolution=1.0, random_seed=SEED)
        from spatial_benchmark.so2_geometry_hl_figures import render_spatial_png
        event(stage, "render_png")
        render_spatial_png(output_root=stage, clustering_receipt=clustering, dpi=300)
        png = "figures/contextual_leiden_resolution_1p0_spatial_14cores.png"
        _atomic_write_text(stage / "README.md", f"# SO2 geometry-modulated hL map\n\n"
            f"Run `{RUN_ID}`, final epoch 200, model seed 0. All {EXPECTED_TOTAL_CELLS:,} cells across 14 fitted cores; "
            f"{clustering['cluster_count']} joint clusters (sizes {clustering['cluster_size_range']}). "
            "Fully observed expression; final graph-block hL, PCA50, cosine kNN30, Leiden resolution1, seed20260825.\n\n"
            "Repeated pilot extraction and seeded Leiden replay passed. Spatial groups can reflect expression, morphology, "
            "core/batch effects or broad fields. This is an exploratory representation map with no cell-type, communication, "
            "patient-generalization or causal claim. No biological null or cross-seed stability was tested.\n\n"
            f"Clusters with >90% membership from one core: {clustering['core_dominated_gt_90pct']}.\n\n"
            "Reproduce from the project root: `PYTHONPATH=src /venv/main/bin/python scripts/analysis/create_so2_geometry_hl_map.py`. "
            "Add `--verify-only` to check the completed report. See TASK_CONTRACT.md and manifests for provenance.\n")
        event(stage, "complete", clusters=clustering["cluster_count"], elapsed_seconds=time.monotonic() - started)
        _atomic_write_json(stage / "status.json", {"status": "complete", "time": now(), "evaluation_id": EVALUATION})
        manifest = _receipt_with_self_hash({"schema": "so2_geometry_hl_report_v1", "status": "complete", "run_id": RUN_ID,
            "evaluation_id": EVALUATION, "provenance": dict(inputs.provenance), "parameters": parameters,
            "total_cells": EXPECTED_TOTAL_CELLS, "cluster_count": clustering["cluster_count"],
            "cluster_size_range": clustering["cluster_size_range"], "core_dominated_gt_90pct": clustering["core_dominated_gt_90pct"],
            "leiden_repeat_identical": determinism["identical_labels"], "png": png,
            "runtime_seconds": time.monotonic() - started, "files": _file_manifest(stage)})
        _atomic_write_json(stage / "manifest.json", manifest)
        _atomic_write_text(stage / "_SUCCESS", sha256_file(stage / "manifest.json") + "\n")
        verify_report(stage)
        final.parent.mkdir(parents=True, exist_ok=True)
        require(not final.exists(), "Report destination already exists")
        os.rename(stage, final)
        verify_report(final)
        register_report(registry, final, manifest)
        print(json.dumps({"status": "complete", "png": str(final / png), "clusters": clustering["cluster_count"]}), flush=True)
    except Exception:
        if stage.exists():
            _atomic_write_json(stage / "status.json", {"status": "failed", "time": now(), "traceback": traceback.format_exc()})
            event(stage, "failed", traceback=traceback.format_exc())
        with registry.transaction() as connection:
            connection.execute("UPDATE evaluations SET status=? WHERE evaluation_id=?", ("failed", EVALUATION))
        raise


if __name__ == "__main__":
    main()

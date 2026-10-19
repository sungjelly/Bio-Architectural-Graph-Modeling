"""Registered CPU-only SO1/SO2 representations and Leiden analysis.

See experiments/campaigns/cmp_20261016_so12_embeddings_leiden/README.md.
The numerical implementation is imported only after CPU thread limits are set.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

from spatial_benchmark.identifiers import (
    canonical_sha256, create_run_id, repro_id, scientific_id, semantic_run_alias,
)
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry, utc_now

CAMPAIGN = "cmp_20261016_so12_embeddings_leiden"
VERSION = "so12_fully_observed_embeddings_v1"
UPSTREAM_RUN = "r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410"
CONFIG = {
    "protocol": VERSION,
    "campaign_id": CAMPAIGN,
    "classification": {"lifecycle_stage": "exploratory_screen", "study_axis": "representation"},
    "model": {"family": "classical_cell_encoders", "embedding_dim": 32},
    "dataset": {
        "dataset_id": "so12_pearson_integration", "version": "v1",
        "split_id": "so12_10train_4diagnostic_donors_v1",
        "preprocessing_version": "so12_quasipoisson_v1",
    },
    "upstream_run_id": UPSTREAM_RUN,
    "methods": ["pca", "kernel_pca", "spatial_pca"],
    "n_components": 32, "kernel_landmarks": 512,
    "kernel": {"type": "rbf", "landmark_selection": "core_balanced_training_only",
               "gamma": "inverse_median_positive_squared_landmark_distance"},
    "spatial_k": 15, "spatial_radius_um": 50.0, "spatial_weight": 0.3,
    "spatial": {"weighting": "gaussian", "sigma": "farthest_eligible_neighbor_distance",
                "isolated_cell": "self_fallback", "boundary": "within_core_and_slide_only",
                "block_scaling": "train_center_and_scalar_rms_each_block"},
    "knn_k": 30,
    "graph": {"neighbor_k": 30, "metric": "euclidean", "backend": "hnsw",
              "M": 32, "ef_construction": 120, "ef_search": 128,
              "ef_search_retry": [256, 512], "exact_recall_queries": 256,
              "minimum_exact_recall": 0.9,
              "directed_weight": "exp(-squared_distance/(kth_squared_distance+epsilon))",
              "symmetrization": "arithmetic_mean"},
    "leiden_resolutions": [0.5, 1.0, 1.5], "leiden_seeds": [0, 1],
    "leiden": {"partition": "RBConfigurationVertexPartition", "n_iterations": 2,
               "second_seed_resolution": 1.0},
    "spatial_null_permutations": 20,
    "train_cores": list(range(1, 11)) + list(range(15, 25)),
    "diagnostic_cores": list(range(11, 15)) + list(range(25, 29)),
    "expected_cells": 407659, "expected_train_cells": 301198,
    "expected_genes": 1000,
    "seed": 0, "fold": 0, "attempt": 1, "threads": 8,
    "claim": "exploratory_fully_observed_transductive_clustering",
    "fit_population": "training_donors_only",
    "clustering_population": "all_mapped_cells",
    "runtime": {"device": "cpu", "minimum_free_gib": 5.0,
                "maximum_bundle_gib": 1.0,
                "leiden_parallel_workers": 4,
                "cpu_only_reason": "protect_active_gpu_training_use_frozen_pca_and_small_kernel"},
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def configure_cpu(threads: int) -> None:
    """Bound BLAS/OpenMP/Numba before importing numerical dependencies."""
    if not 1 <= threads <= 8:
        raise ValueError("--threads must be between 1 and 8")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["MPLBACKEND"] = "Agg"


def environment_snapshot(threads: int) -> dict:
    packages = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn",
                 "faiss-cpu", "igraph", "leidenalg", "numba", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    from threadpoolctl import threadpool_info
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "host": socket.gethostname(), "cpu_count": os.cpu_count(),
        "cpu_threads": threads, "device": "cpu", "gpu_device_ids": [],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_usage": "none; active training GPUs protected",
        "packages": packages, "thread_pools": threadpool_info(),
        "environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MPLBACKEND")},
    }


def verify_inputs(paths, reg: Registry, config: dict) -> dict:
    """Reuse verified registered inputs; never recreate or mutate registrations."""
    spec = config["dataset"]
    dataset = reg.get_dataset(spec["dataset_id"], spec["version"])
    split = reg.get_split(spec["split_id"])
    upstream = reg.get_run(config["upstream_run_id"])
    if not dataset or dataset["verification_status"] != "verified":
        raise ValueError("Expected verified SO12 Pearson dataset registration")
    if not split or split["verification_status"] != "verified":
        raise ValueError("Expected verified donor split registration")
    if split["dataset_id"] != spec["dataset_id"] or split["dataset_version"] != spec["version"]:
        raise ValueError("Split registration belongs to a different dataset")
    for key in ("train_cores", "diagnostic_cores"):
        if sorted(split["metadata"][key]) != sorted(config[key]):
            raise ValueError(f"Registered {key} disagree with fixed configuration")
    if set(config["train_cores"]) & set(config["diagnostic_cores"]):
        raise ValueError("Training and diagnostic cores overlap")
    if not upstream or upstream["status"] != "completed":
        raise ValueError("Upstream preprocessing run is not completed")
    source = paths.data_root / "processed/so12_pearson_integration_v1"
    manifest_path = source / "manifest.json"
    manifest_sha = sha(manifest_path)
    if manifest_sha != dataset["processed_fingerprint"]:
        raise ValueError("Dataset manifest checksum differs from registration")
    manifest = json.loads(manifest_path.read_text())
    if manifest["run_id"] != config["upstream_run_id"]:
        raise ValueError("Unexpected upstream preprocessing identity")
    if manifest["cells"] != config["expected_cells"] or manifest["genes"] != config["expected_genes"]:
        raise ValueError("Unexpected dataset dimensions")
    upstream_dir = Path(upstream["artifact_path"])
    if not (upstream_dir / "_SUCCESS").is_file():
        raise ValueError("Upstream successful bundle marker is missing")
    upstream_checksums = json.loads((upstream_dir / "checksums.json").read_text())
    pca_relative = "methods/pearson/fitted_parameters.npz"
    pca_source = upstream_dir / pca_relative
    pca_sha = sha(pca_source)
    if upstream_checksums.get(pca_relative) != pca_sha:
        raise ValueError("Upstream fitted PCA parameters do not match their checksum")
    if sha(source / "pearson/fitted_parameters.npz") != pca_sha:
        raise ValueError("Dataset fitted PCA parameters differ from upstream run")
    files = {"manifest.json": manifest_sha}
    for relative in ("observations.csv.gz", "genes.json", "pearson/fitted_parameters.npz"):
        files[relative] = sha(source / relative)
    # Array checksums and alignment are verified by the numerical loader before use.
    return {
        "dataset_id": spec["dataset_id"], "dataset_version": spec["version"],
        "dataset_root": str(source), "dataset_fingerprint": manifest_sha,
        "split_id": spec["split_id"], "split_fingerprint": split["fingerprint"],
        "split": split, "source_file_sha256": files,
        "upstream_run_id": config["upstream_run_id"],
        "upstream_artifact_path": str(upstream_dir),
        "upstream_pca_sha256": pca_sha,
        "upstream_checksums_sha256": sha(upstream_dir / "checksums.json"),
        "source_arrays": [row for row in manifest["matrices"] if row["method"] == "pearson"],
        "permitted_claim": config["claim"],
    }


class Tee:
    """Retain runtime logs in the bundle while streaming to supervisor."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def metric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from metric_leaves(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from metric_leaves(item, f"{prefix}.{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield prefix, float(value)


def verify_outputs(out: Path, config: dict) -> dict:
    """Independently reopen saved representations, labels, and portable report."""
    import numpy as np
    import pandas as pd

    n = config["expected_cells"]
    rows = np.arange(n)
    cells = pd.read_csv(out / "cells.csv.gz")
    if len(cells) != n or not np.array_equal(cells.row_index.to_numpy(), rows):
        raise ValueError("Saved observation rows are incomplete or reordered")
    if cells.duplicated(["slide", "fov", "cell_ID"]).any():
        raise ValueError("Saved observation keys are not unique")
    required_labels = {"row_index"} | {
        f"leiden_r{resolution:.1f}_s{seed}"
        for resolution in config["leiden_resolutions"]
        for seed in (config["leiden_seeds"] if resolution == 1.0 else [config["seed"]])
    }
    methods = {}
    for method in config["methods"]:
        folder = out / "methods" / method
        embedding = np.load(folder / "embedding.npy", mmap_mode="r", allow_pickle=False)
        if embedding.shape != (n, config["n_components"]) or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid saved embedding shape/values for {method}")
        clusters = pd.read_csv(folder / "clusters.csv.gz")
        if set(clusters.columns) != required_labels or len(clusters) != n:
            raise ValueError(f"Missing or unexpected saved Leiden labels for {method}")
        if not np.array_equal(clusters.row_index.to_numpy(), rows):
            raise ValueError(f"Saved cluster rows are not aligned for {method}")
        for column in sorted(required_labels - {"row_index"}):
            values = clusters[column].to_numpy()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid saved labels: {method}/{column}")
            if not np.array_equal(values, values.astype(np.int64)):
                raise ValueError(f"Noninteger saved labels: {method}/{column}")
        methods[method] = {"embedding_shape": list(embedding.shape), "all_finite": True,
                           "labels_aligned": True, "label_columns": sorted(required_labels - {"row_index"})}

    class PortableImages(HTMLParser):
        def __init__(self):
            super().__init__()
            self.images = 0

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag.lower() == "img":
                self.images += 1
                if not (attributes.get("src") or "").startswith("data:image/"):
                    raise ValueError("Report image is not embedded as a portable data URL")
                if attributes.get("srcset"):
                    raise ValueError("Report must use a single embedded source per image")

    report = out / "report/index.html"
    content = report.read_text()
    parser = PortableImages()
    parser.feed(content)
    if not parser.images or re.search(r"url\s*\(\s*['\"]?(?:https?:|//)", content, re.IGNORECASE):
        raise ValueError("Report has no embedded images or includes external CSS image sources")
    return {"status": "passed", "cells": n, "slide_qualified_keys_unique": True,
            "methods": methods, "portable_report": str(report.relative_to(out)),
            "embedded_report_images": parser.images}


def bundle_inventory(folder: Path) -> dict:
    return {str(path.relative_to(folder)): {
        "sha256": sha(path), "size_bytes": path.stat().st_size,
    } for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != "checksums.json"}


def publish(out: Path, paths, reg: Registry, run_id: str, status: str, started: float) -> Path:
    """Finalize log-closed files and atomically move the bundle, without copying."""
    marker = "_SUCCESS" if status == "completed" else "_FAILED"
    opposite = out / ("_FAILED" if status == "completed" else "_SUCCESS")
    if opposite.exists():
        opposite.unlink()
    (out / marker).write_text(run_id + "\n")
    write_json(out / "status.json", {
        "run_id": run_id, "status": status, "end_time": utc_now(),
        "duration_seconds": time.monotonic() - started,
    })
    write_json(out / "checksums.json", bundle_inventory(out))
    partition = datetime.now(timezone.utc).strftime("%Y/%m")
    final = paths.artifact_root / "runs" / partition / run_id
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite run bundle: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    out.rename(final)
    reg.record_artifacts(run_id, [
        {"kind": "representation_analysis_evidence", "path": path,
         "sha256": sha(path), "size_bytes": path.stat().st_size,
         "status": "verified" if status == "completed" else "present"}
        for path in sorted(final.rglob("*")) if path.is_file()
    ])
    updates = {"artifact_path": final, "end_time": utc_now(),
               "duration_seconds": time.monotonic() - started}
    if status == "failed":
        updates["failure_category"] = "representation_analysis_exception"
    reg.transition_run(run_id, status, **updates)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Synthetic controls only; no registered full run")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--retry-of")
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("--attempt must be positive")
    config = deepcopy(CONFIG)
    config.update(threads=args.threads, attempt=args.attempt)
    configure_cpu(args.threads)
    from so12_embedding_methods import run_analysis, smoke
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        controls = smoke(config)
        if args.smoke:
            print(json.dumps(controls, indent=2, allow_nan=False))
            return
        paths = current_paths()
        reg = Registry()
        paths.scratch_root.mkdir(parents=True, exist_ok=True)
        paths.artifact_root.mkdir(parents=True, exist_ok=True)
        minimum = config["runtime"]["minimum_free_gib"]
        maximum = config["runtime"]["maximum_bundle_gib"]
        for folder in (paths.scratch_root, paths.artifact_root):
            if shutil.disk_usage(folder).free < (minimum + maximum) * 2**30:
                raise RuntimeError(f"Need {minimum + maximum:g} GiB free on {folder} before analysis")
        if paths.scratch_root.stat().st_dev != paths.artifact_root.stat().st_dev:
            raise RuntimeError("Scratch and artifact roots must share a filesystem for atomic publication")
        provenance = verify_inputs(paths, reg, config)
        source_files = [paths.project_root / relative for relative in (
            "scripts/analysis/run_so12_embeddings.py", "scripts/analysis/so12_embedding_methods.py",
            "scripts/analysis/so12_embedding_report.py",
            "ops/supervisor/bagm-so12-embeddings.sh", "ops/supervisor/bagm-so12-embeddings.conf",
            "src/spatial_benchmark/paths.py", "src/spatial_benchmark/identifiers.py",
            "src/spatial_benchmark/registry.py",
        )]
        source_hashes = {str(path.relative_to(paths.project_root)): sha(path) for path in source_files}
        environment = environment_snapshot(args.threads)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.project_root, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=paths.project_root, text=True)
        diff = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=paths.project_root)
        fingerprint = canonical_sha256({"sources": source_hashes,
                                       "git_diff_sha256": hashlib.sha256(diff).hexdigest()})
        sci = scientific_id(config)
        rep = repro_id(config, git_commit=commit, dirty_fingerprint=fingerprint,
                       dataset_fingerprint=provenance["dataset_fingerprint"],
                       split_fingerprint=provenance["split_fingerprint"],
                       preprocessing_version=config["dataset"]["preprocessing_version"],
                       environment_fingerprint=canonical_sha256(environment))
        run_id = create_run_id(seed=config["seed"], fold=config["fold"], attempt=args.attempt, scientific_id_value=sci)
        out = paths.scratch_root / "active_runs" / run_id
        out.mkdir(parents=True, exist_ok=False)
        existing_campaign = reg.get_campaign(CAMPAIGN)
        reg.create_campaign(CAMPAIGN, name="SO1 SO2 normalized representations and Leiden",
                            scientific_question="How do linear, nonlinear, and local spatial representations change exploratory cell partitions?",
                            config=existing_campaign["config"] if existing_campaign else CONFIG,
                            status="exploratory")
        reg.register_variant(sci, campaign_id=CAMPAIGN, configuration=config)
        reg.create_run(run_id, campaign_id=CAMPAIGN, scientific_id=sci, repro_id=rep,
                       seed=config["seed"], fold=config["fold"], attempt=args.attempt,
                       configuration=config, artifact_path=out, retry_of=args.retry_of,
                       git_commit=commit, dirty_status=bool(git_status),
                       preprocessing_version=config["dataset"]["preprocessing_version"],
                       dataset_fingerprint=provenance["dataset_fingerprint"],
                       split_fingerprint=provenance["split_fingerprint"],
                       host=socket.gethostname(), gpu_model="CPU only", peak_vram_gb=0.0)
        started = time.monotonic()
        try:
            alias = semantic_run_alias(primary_run_id=run_id, historical=False, categories={
                "lifecycle_stage": "exploratory_screen", "study_axis": "representation",
                "model": "three-encoders", "graph": "knn30", "mask": "none",
                "edge_feature_state": "distance", "embedding_dim": 32,
                "seed": config["seed"], "fold": config["fold"], "attempt": args.attempt,
                "variant_evidence": {"scientific_id": sci},
            })
            reg.register_run_semantics(run_id, lifecycle_stage="exploratory_screen", study_axis="representation",
                source_batch="so12_joint", seed_known=True, fold_known=True, attempt_known=True,
                retention_class="analysis_evidence", category_key="representations.leiden",
                classification_confidence="high", timestamp_basis="execution",
                variant_label="pca_kernel_spatial", model_key="classical_cell_encoders",
                dataset_key="so12_28core", masking_key="none", graph_key="expression_knn30",
                feature_key="pearson_biological_1000", embedding_key="three_encoders_32",
                semantic_alias=alias, preferred_alias_type="semantic")
            reg.transition_run(run_id, "running", start_time=utc_now())
            write_json(out / "resolved_config.json", config)
            write_json(out / "environment.json", environment)
            write_json(out / "smoke_checks.json", controls)
            write_json(out / "input_provenance.json", provenance)
            write_json(out / "source_code_sha256.json", source_hashes)
            write_json(out / "identity.json", {
                "run_id": run_id, "semantic_alias": alias, "scientific_id": sci, "repro_id": rep,
                "git_commit": commit, "dirty_fingerprint": fingerprint, "command": sys.argv,
                "cwd": str(Path.cwd()), "project_root": str(paths.project_root),
                "checkpoint": "not_applicable_non_neural_representation_analysis",
            })
            (out / "git_status.txt").write_text(git_status)
            (out / "git_diff.patch").write_bytes(diff)
            for source in source_files:
                dest = out / "code" / source.relative_to(paths.project_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
            shutil.copy2(paths.project_root / "experiments/campaigns" / CAMPAIGN / "README.md", out / "task_contract.md")
            with (out / "execution.log").open("a", buffering=1) as log:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                    print(f"RUN_ID={run_id} CPU_THREADS={args.threads}", flush=True)
                    try:
                        summary = run_analysis(out, config)
                        if not isinstance(summary, dict):
                            raise TypeError("run_analysis must return a JSON-serializable dictionary")
                        output_verification = verify_outputs(out, config)
                        write_json(out / "output_verification.json", output_verification)
                        write_json(out / "run_summary.json", summary)
                        with (out / "metrics.jsonl").open("a") as metrics:
                            for name, value in metric_leaves(summary):
                                metrics.write(json.dumps({"name": name, "value": value,
                                    "split": "all_cells_transductive", "recorded_at": utc_now()}) + "\n")
                                reg.record_metric(run_id, name, value, split="all_cells_transductive")
                    except BaseException:
                        traceback.print_exc()
                        raise
            output_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
            if output_bytes > maximum * 2**30:
                raise RuntimeError(f"Bundle exceeds fixed {maximum:g}-GiB storage budget")
            if shutil.disk_usage(out).free < minimum * 2**30:
                raise RuntimeError(f"Free storage below {minimum:g}-GiB reserve")
            write_json(out / "runner_verification.json", {
                "status": "passed", "registered_dataset_identity": True,
                "registered_split_identity": True, "upstream_pca_checksum": True,
                "output_bytes_before_inventory": output_bytes,
                "runtime_seconds": time.monotonic() - started,
                "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                "peak_child_process_rss_gib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20,
                "host_memory_metric_note": "Self and largest completed child peaks are separate; child value is not concurrent aggregate RSS",
                "peak_vram_gib": 0.0, "free_disk_gib": shutil.disk_usage(out).free / 2**30,
                "checkpoint": "not_applicable", "source_code_sha256": source_hashes,
            })
            reg.transition_run(run_id, "finalizing")
            final = publish(out, paths, reg, run_id, "completed", started)
        except BaseException:
            if out.exists():
                (out / "failure.txt").write_text(traceback.format_exc())
                final = publish(out, paths, reg, run_id, "failed", started)
                print(f"FAILED {final}", file=sys.stderr, flush=True)
            else:
                # Publication may have succeeded before registry I/O failed.
                # Retain the checksum-bound bundle unchanged for explicit recovery.
                print(f"Registry finalization interrupted for {run_id}; inspect published bundle", file=sys.stderr, flush=True)
            raise
        print(f"COMPLETED {final}", flush=True)


if __name__ == "__main__":
    main()

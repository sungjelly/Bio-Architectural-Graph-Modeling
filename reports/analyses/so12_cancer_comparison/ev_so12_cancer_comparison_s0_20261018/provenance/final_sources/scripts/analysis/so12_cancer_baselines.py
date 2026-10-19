#!/usr/bin/env python3
"""Train-only PCA/scVI comparators for the frozen SO12 paper test population.

The managed caller owns GPU locks and publication. This module never fits on
test cells, writes source data, or changes the previously published cell index.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import so12_paper_test_extraction as base
from so12_paper_test_h0_extraction import verify_source
from scripts.analysis.run_so12_paper_test_leiden import verify_published
from spatial_benchmark.paths import current_paths
from spatial_benchmark.so12_paper_split_data import (
    TRAIN_CORE_NUMBERS, VALIDATION_CORE_NUMBERS, ROLE_CELLS, load_so12_paper_data,
)

DIMENSIONS = 50
SEED = 0
REFERENCE_EVALUATION = "ev_so12_paper_test_hL_s0_20261017"
SCVI_CONFIG = {
    "seed": SEED, "n_latent": DIMENSIONS, "n_hidden": 128, "n_layers": 2,
    "gene_likelihood": "nb", "dispersion": "gene", "dropout_rate": 0.1,
    "use_observed_lib_size": True, "latent_distribution": "normal",
    "encode_covariates": False, "batch_key": "slide", "batch_size": 1024,
    "max_epochs": 100, "early_stopping_patience": 15,
    "early_stopping_monitor": "elbo_validation", "early_stopping_min_delta": 0.0,
    "n_epochs_kl_warmup": 20, "early_stopping_warmup_epochs": 20,
    "learning_rate": 0.001, "test_query_adaptation": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path: Path, value: Any):
    base._write(Path(path), value)


def disk_gate(path: Path):
    require(shutil.disk_usage(path).free >= 5 * 2**30, "The 5 GiB disk reserve was reached.")


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(array_hash(tensor.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def verify_membership(cores, roles, expected_counts=ROLE_CELLS):
    expected = {"train": TRAIN_CORE_NUMBERS, "validation": VALIDATION_CORE_NUMBERS,
                "test": (15, 23)}
    require(all(core.split in roles for core in cores), "Unexpected role in baseline data.")
    for role in roles:
        selected = [core for core in cores if core.split == role]
        require(tuple(int(core.alias[-2:]) for core in selected) == expected[role],
                f"The {role} core membership/order changed.")
        require(sum(core.n_cells for core in selected) == expected_counts[role],
                f"The {role} cell count changed.")


def fit_pca_training(cores, n_components=DIMENSIONS, chunk_rows=4096):
    """Centered FP64 covariance using only train roles; no validation/test access."""
    selected = [core for core in cores if core.split == "train"]
    require(bool(selected), "PCA requires training cells.")
    n, mean, m2 = 0, None, None
    for core in selected:
        matrix = core.target_expression
        require(matrix.ndim == 2, "PCA features must be a matrix.")
        for start in range(0, len(matrix), chunk_rows):
            values = np.asarray(matrix[start:start + chunk_rows], dtype=np.float64)
            require(np.isfinite(values).all(), "Nonfinite PCA training features.")
            batch_n = len(values)
            batch_mean = values.mean(axis=0)
            centered = values - batch_mean
            batch_m2 = centered.T @ centered
            if mean is None:
                mean, m2, n = batch_mean, batch_m2, batch_n
            else:
                require(mean.shape == batch_mean.shape, "PCA gene dimension changed.")
                delta = batch_mean - mean
                total = n + batch_n
                m2 += batch_m2 + np.outer(delta, delta) * (n * batch_n / total)
                mean += delta * (batch_n / total)
                n = total
    require(n > n_components and 0 < n_components <= len(mean), "Invalid PCA rank.")
    covariance = m2 / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, order].T.copy()
    # Fix the arbitrary sign so repeated runs yield the same stored projection.
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(n_components), pivots])
    components *= np.where(signs == 0, 1, signs)[:, None]
    selected_values = np.maximum(eigenvalues[order], 0)
    total_variance = np.maximum(eigenvalues, 0).sum()
    require(total_variance > 0, "PCA training data have no variance.")
    return {"mean": mean, "components": components,
            "explained_variance": selected_values,
            "explained_variance_ratio": selected_values / total_variance,
            "training_cells": np.asarray(n, dtype=np.int64)}


def project_pca(matrix, fitted, chunk_rows=4096):
    require(matrix.ndim == 2 and matrix.shape[1] == len(fitted["mean"]), "PCA gene order/shape mismatch.")
    output = np.empty((len(matrix), len(fitted["components"])), dtype=np.float32)
    for start in range(0, len(matrix), chunk_rows):
        values = np.asarray(matrix[start:start + chunk_rows], dtype=np.float64)
        output[start:start + len(values)] = (values - fitted["mean"]) @ fitted["components"].T
    require(np.isfinite(output).all(), "Nonfinite projected PCA embeddings.")
    return output


def count_inputs(cores, genes):
    """Return raw counts, slide codes and explicit indices, excluding test roles."""
    require(bool(cores) and all(core.split in {"train", "validation"} for core in cores),
            "scVI fitting input cannot include test cells.")
    require(len(genes) == len(set(genes)), "Duplicate gene names.")
    matrices, slides, training, validation = [], [], [], []
    offset = 0
    for core in cores:
        values = core.raw_counts
        require(values.shape == (core.n_cells, len(genes)) and np.issubdtype(values.dtype, np.integer)
                and np.all(values >= 0), "scVI requires aligned nonnegative integer raw counts.")
        matrices.append(values)
        slides.extend([str(core.slide)] * core.n_cells)
        indices = np.arange(offset, offset + core.n_cells, dtype=np.int64)
        (training if core.split == "train" else validation).append(indices)
        offset += core.n_cells
    train = np.concatenate(training)
    val = np.concatenate(validation)
    require(set(np.asarray(slides)[train]) == {"1", "2"}, "Both slide batches must occur in training.")
    return np.concatenate(matrices).astype(np.float32), np.asarray(slides), [train, val, np.array([], dtype=np.int64)]


def validate_test_alignment(dataset, cells, expected_genes):
    base._check_test_dataset(dataset)
    require(tuple(dataset.genes) == tuple(expected_genes), "Test gene order differs from training.")
    for core in dataset.cores:
        rows = cells.loc[cells.alias == core.alias]
        require(len(rows) == core.n_cells and rows.slide.eq(core.slide).all()
                and np.allclose(rows[["x_um", "y_um"]].to_numpy(), core.coordinates_um, rtol=0, atol=1e-9),
                "Test count rows differ from the published canonical cell order.")


class BestValidationState:
    """Small CPU copy of the best state, with no per-epoch checkpoint files."""
    def __init__(self):
        self.score, self.epoch, self.state = float("inf"), None, None

    def observe(self, score, epoch, state):
        score = float(score)
        require(np.isfinite(score), "Nonfinite scVI validation ELBO.")
        if score < self.score:
            self.score, self.epoch = score, int(epoch)
            self.state = {name: tensor.detach().cpu().clone() for name, tensor in state.items()}
            return True
        return False


def prepare_context(output, paths):
    output = Path(output).resolve()
    active = (paths.scratch_root / "active_runs").resolve()
    require(output.is_relative_to(active) and output != active, "Baseline output must be an owned active run.")
    output.mkdir(parents=True, exist_ok=True)
    disk_gate(output)
    source_path = paths.artifact_root / "runs/2026/10" / base.SOURCE_RUN_ID
    source = verify_source(source_path, paths)
    reference = paths.report_root / "analyses/so12_paper_test_leiden" / REFERENCE_EVALUATION
    reference_config = base._read(reference / "config.resolved.json")
    manifest = verify_published(reference, reference_config)
    receipt = manifest["summary"]["extraction"]
    require(receipt["source_run_id"] == base.SOURCE_RUN_ID
            and receipt["checkpoint_sha256"] == source["checkpoint_sha256"]
            and receipt["fitted_parameters_sha256"] == source["fitted_parameters_sha256"],
            "Baseline reference differs from the frozen source.")
    cells = pd.read_csv(reference / "cells.csv.gz", float_precision="round_trip")
    require(base._row_digest(cells) == receipt["cell_rows_sha256"], "Published cell index changed.")
    reference_info = {"evaluation_id": REFERENCE_EVALUATION, "path": str(reference),
                      "manifest_sha256": base.file_hash(reference / "manifest.json"),
                      "cells_sha256": base.file_hash(reference / "cells.csv.gz"),
                      "cell_rows_sha256": receipt["cell_rows_sha256"]}
    return output, source, cells, reference_info


def load_data(source, paths, roles):
    dataset = load_so12_paper_data(source["config"], paths, roles=roles,
                                   verify_checksums=True, load_graphs=False)
    verify_membership(dataset.cores, roles)
    for key in ("dataset_fingerprint", "split_fingerprint", "preprocessing_fingerprint", "fitted_parameters_sha256"):
        require(dataset.provenance[key] == source[key], f"Baseline {key} differs from frozen source.")
    return dataset


def run_pca(output, source, cells, reference, paths):
    started = time.monotonic()
    train = load_data(source, paths, ("train",))
    genes = train.genes
    fitted = fit_pca_training(train.cores)
    require(int(fitted["training_cells"]) == ROLE_CELLS["train"], "PCA fit population changed.")
    model_path = output / "baselines/pca_model.npz"
    require(not model_path.exists(), "Refusing to overwrite fitted PCA.")
    model_path.parent.mkdir(exist_ok=True)
    np.savez(model_path, **fitted, genes=np.asarray(genes))
    fit_receipt = {"method": "pca", "fitted_model_sha256": base.file_hash(model_path),
                   "training_cells": ROLE_CELLS["train"], "training_core_numbers": list(TRAIN_CORE_NUMBERS),
                   "validation_used_for_fitting": False, "test_used_for_fitting": False,
                   "fit_elapsed_seconds": time.monotonic() - started}
    write(output / "baselines/pca_frozen_before_test.json", fit_receipt)
    del train
    gc.collect()
    test = load_data(source, paths, ("test",))
    validate_test_alignment(test, cells, genes)
    embedding = np.concatenate([project_pca(core.target_expression, fitted) for core in test.cores])
    require(base.file_hash(model_path) == fit_receipt["fitted_model_sha256"], "PCA changed during projection.")
    return embedding, {**fit_receipt, "representation": "training_fitted_frozen_pearson_pca50",
                       "explained_variance_ratio": fitted["explained_variance_ratio"].tolist(),
                       "centering": "training_only", "whiten": False}, genes


def run_scvi(output, source, cells, reference, paths, device):
    import anndata
    import scvi
    import torch
    from lightning.pytorch.callbacks import Callback

    require(scvi.__version__ == "1.5.1", "The approved scVI package version changed.")
    started = time.monotonic()
    scvi.settings.seed = SEED
    torch.set_num_threads(4)
    base.configure_deterministic_inference()
    trainval = load_data(source, paths, ("train", "validation"))
    genes = trainval.genes
    counts, slides, indexing = count_inputs(trainval.cores, genes)
    adata = anndata.AnnData(X=counts, obs=pd.DataFrame({"slide": pd.Categorical(slides, categories=["1", "2"])}),
                           var=pd.DataFrame(index=list(genes)))
    scvi.model.SCVI.setup_anndata(adata, batch_key="slide")
    model = scvi.model.SCVI(adata, n_latent=DIMENSIONS, n_hidden=128, n_layers=2,
                           gene_likelihood="nb", dispersion="gene", dropout_rate=0.1,
                           use_observed_lib_size=True, latent_distribution="normal", encode_covariates=False)
    best = BestValidationState()
    progress = output / "baselines/scvi_training.jsonl"
    epoch_start = [time.monotonic()]

    class CaptureBest(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            epoch_start[0] = time.monotonic()

        def on_validation_end(self, trainer, pl_module):
            require("elbo_validation" in trainer.callback_metrics, "Missing validation ELBO.")
            score = float(trainer.callback_metrics["elbo_validation"].detach().cpu())
            improved = best.observe(score, trainer.current_epoch, model.module.state_dict())
            disk_gate(output)
            row = {"epoch": int(trainer.current_epoch), "elbo_validation": score,
                   "best_epoch": best.epoch, "best_elbo_validation": best.score,
                   "improved": improved, "epoch_seconds": time.monotonic() - epoch_start[0],
                   "elapsed_seconds": time.monotonic() - started}
            with progress.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            print(json.dumps({"stage": "scvi_training", **row}), flush=True)
            if trainer.current_epoch == 1:
                write(output / "baselines/scvi_runtime_pilot.json", {
                    **row, "max_epochs": 100, "approximate_max_training_seconds": row["epoch_seconds"] * 100,
                    "estimate_excludes_preparation_and_may_end_early": True})

    baseline_dir = output / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    write(baseline_dir / "scvi_training_plan.json", {**SCVI_CONFIG,
          "train_indices_sha256": array_hash(indexing[0]), "validation_indices_sha256": array_hash(indexing[1]),
          "training_core_numbers": list(TRAIN_CORE_NUMBERS), "validation_core_numbers": list(VALIDATION_CORE_NUMBERS),
          "training_cells": len(indexing[0]), "validation_cells": len(indexing[1]), "test_cells_in_fit": 0})
    accelerator = "gpu" if device.startswith("cuda") else "cpu"
    devices = [int(device.split(":")[1])] if accelerator == "gpu" and ":" in device else 1
    model.train(max_epochs=SCVI_CONFIG["max_epochs"], accelerator=accelerator, devices=devices,
                batch_size=SCVI_CONFIG["batch_size"],
                datasplitter_kwargs={"external_indexing": indexing},
                early_stopping=True, early_stopping_monitor="elbo_validation",
                early_stopping_patience=SCVI_CONFIG["early_stopping_patience"], early_stopping_min_delta=0.0,
                early_stopping_warmup_epochs=SCVI_CONFIG["early_stopping_warmup_epochs"], check_val_every_n_epoch=1,
                plan_kwargs={"lr": SCVI_CONFIG["learning_rate"], "n_epochs_kl_warmup": SCVI_CONFIG["n_epochs_kl_warmup"]},
                callbacks=[CaptureBest()], enable_checkpointing=False, logger=False,
                enable_progress_bar=False, deterministic=True, benchmark=False,
                default_root_dir=str(baseline_dir), num_sanity_val_steps=0)
    require(best.state is not None, "No finite validation-selected scVI checkpoint.")
    model.module.load_state_dict(best.state, strict=True)
    model.module.eval()
    selected_state_hash = state_hash(model.module.state_dict())
    require(selected_state_hash == state_hash(best.state), "Best scVI state failed restoration.")
    model_dir = baseline_dir / "scvi_model"
    require(not model_dir.exists(), "Refusing to overwrite fitted scVI.")
    model.save(model_dir, overwrite=False, save_anndata=False)
    model_file = model_dir / "model.pt"
    require(model_file.is_file() and not list(model_dir.glob("*.h5ad")), "scVI compact checkpoint contract failed.")
    freeze = {"method": "scvi", "configuration": SCVI_CONFIG,
              "selected_epoch_zero_based": best.epoch, "validation_elbo": best.score,
              "model_sha256": base.file_hash(model_file), "state_sha256": selected_state_hash,
              "training_cells": ROLE_CELLS["train"], "validation_cells": ROLE_CELLS["validation"],
              "training_core_numbers": list(TRAIN_CORE_NUMBERS), "validation_core_numbers": list(VALIDATION_CORE_NUMBERS),
              "test_used_for_fitting": False, "test_query_adaptation": False,
              "fit_elapsed_seconds": time.monotonic() - started}
    write(baseline_dir / "scvi_frozen_before_test.json", freeze)
    del trainval, counts, slides, best
    gc.collect()
    test = load_data(source, paths, ("test",))
    validate_test_alignment(test, cells, genes)
    query = anndata.AnnData(X=np.concatenate([core.raw_counts for core in test.cores]).astype(np.float32),
                           obs=pd.DataFrame({"slide": pd.Categorical([str(core.slide) for core in test.cores
                                                   for _ in range(core.n_cells)], categories=["1", "2"])}),
                           var=pd.DataFrame(index=list(genes)))
    require(set(query.obs.slide).issubset(set(adata.obs.slide)), "Unseen scVI query batch category.")
    embedding = np.asarray(model.get_latent_representation(query, give_mean=True, batch_size=1024), dtype=np.float32)
    require(state_hash(model.module.state_dict()) == selected_state_hash
            and base.file_hash(model_file) == freeze["model_sha256"], "scVI weights changed during test inference.")
    return embedding, {**freeze, "representation": "scvi50_raw_count_slide_batch_integrated_posterior_mean",
                       "scvi_version": scvi.__version__, "batch_categories": ["1", "2"],
                       "information_difference": "expression_only_with_slide_covariate_no_morphology_or_spatial_graph"}, genes


def run(output: Path, mode: str, device="cuda:0", paths=None):
    require(mode in {"pca", "scvi"}, "Unsupported baseline mode.")
    paths = paths or current_paths()
    output = Path(output).resolve()
    require(not (output / f"{mode}_manifest.json").exists()
            and not (output / f"embeddings/{mode}.npy").exists(), "Refusing to overwrite baseline outputs.")
    started = time.monotonic()
    output, source, cells, reference = prepare_context(output, paths)
    (output / "baselines").mkdir(exist_ok=True)
    (output / "embeddings").mkdir(exist_ok=True)
    embedding, details, genes = (run_pca(output, source, cells, reference, paths) if mode == "pca"
                                else run_scvi(output, source, cells, reference, paths, device))
    require(embedding.shape == (base.N_CELLS, DIMENSIONS) and embedding.dtype == np.float32
            and np.isfinite(embedding).all() and np.all(np.linalg.norm(embedding, axis=1) > 0),
            "Baseline embeddings have missing, zero or invalid rows.")
    path = output / f"embeddings/{mode}.npy"
    np.save(path, embedding, allow_pickle=False)
    require(base.training_source_binding() == source["binding"]["source_sha256"], "Frozen training source changed.")
    require(base.file_hash(Path(reference["path"]) / "manifest.json") == reference["manifest_sha256"],
            "Published reference changed.")
    disk_gate(output)
    result = {"schema": "so12_cancer_baseline_v1", "status": "complete", "passed": True,
              "representation_name": mode, "source_run_id": base.SOURCE_RUN_ID,
              "checkpoint_epoch": 194, "checkpoint_sha256": source["checkpoint_sha256"],
              "fitted_parameters_sha256": source["fitted_parameters_sha256"],
              "preprocessing_fingerprint": source["preprocessing_fingerprint"],
              "reference_hl": reference, "cell_rows_sha256": reference["cell_rows_sha256"],
              "cells_sha256": reference["cells_sha256"], "genes_sha256": base.digest(list(genes)),
              "n_genes": len(genes), "dimensions": DIMENSIONS,
              "embedding_path": str(path.relative_to(output)), "embedding_sha256": base.file_hash(path),
              "n_cells": base.N_CELLS, "test_core_numbers": [15, 23],
              "cells_per_core": {str(key): value for key, value in base.CORE_CELLS.items()},
              "test_only_output": True, "source_modified": False, "normalization_refit": False,
              "analysis_source_sha256": {"scripts/analysis/so12_cancer_baselines.py": base.file_hash(Path(__file__))},
              "elapsed_seconds": time.monotonic() - started,
              "created_at": datetime.now(timezone.utc).isoformat(), **details}
    write(output / f"{mode}_manifest.json", result)
    print(json.dumps({"stage": mode + "_complete", "elapsed_seconds": result["elapsed_seconds"],
                      "embedding_sha256": result["embedding_sha256"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pca", "scvi"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.output, args.mode, args.device)


if __name__ == "__main__":
    main()

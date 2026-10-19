#!/usr/bin/env python3
"""Queue-owned SO2 GeneMAE comparison. See the campaign README before running."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spatial_benchmark.configuration import load_yaml_mapping
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.myjju_genemae import (
    make_source_model, count_parameters, recursive_spatial_tiles,
    build_symmetric_knn_graph, permute_graph_node_labels,
)
from spatial_benchmark.paths import current_paths
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle
from spatial_benchmark.so2_genemae_masking import (
    ALIASES, METHODS, PRIMARY, ReconstructionMetrics, array_digest,
    make_mask, neighbor_mean, resource_gates, runtime_path, stable_seed, validate_protocol, visible_mean,
)


def load_inputs(config, archive):
    root = runtime_path(config["dataset"]["prepared_artifact_reference"])
    if sha256_file(root / "manifest.json") != config["dataset"]["dataset_fingerprint"]:
        raise ValueError("Cohort manifest drift")
    manifest = json.loads((root / "manifest.json").read_text())
    signed = dict(manifest)
    checksum = signed.pop("manifest_content_sha256")
    if canonical_sha256(signed) != checksum:
        raise ValueError("Invalid cohort manifest content hash")
    if (manifest["cohort"]["aliases"] != list(ALIASES)
        or manifest["cohort"]["total_cells"] != 246063
        or not manifest["assurances"]["unmapped_fov_246_excluded"]):
        raise ValueError("SO2 cohort/routing drift")
    genes = manifest["features"]["gene_names"]
    if len(genes) != 1000 or len(set(genes)) != 1000 or any(
        x.startswith(("Negative", "SystemControl")) for x in genes):
        raise ValueError("Invalid biological panel")
    cores, provenance = [], []
    for alias in ALIASES:
        relative = f"cores/{alias}.npz"
        if sha256_file(root / relative) != manifest["files"][relative]:
            raise ValueError(f"Input checksum mismatch: {alias}")
        with np.load(root / relative, allow_pickle=False) as z:
            raw = z["expression_counts"]
            coordinates = z["coordinates_um"]
            if raw.shape[1] != 1000 or not np.isfinite(raw).all() or raw.min() < 0:
                raise ValueError("Invalid expression")
            target = np.log1p(raw).astype(np.float32)
        if coordinates.shape != (len(target), 2) or not np.isfinite(coordinates).all():
            raise ValueError("Coordinate alignment failure")
        tiles = []
        for index, nodes in enumerate(recursive_spatial_tiles(coordinates, max_nodes=7000)):
            edges, attrs = build_symmetric_knn_graph(coordinates[nodes], k=15)
            if (edges[0] == edges[1]).any() or edges.min() < 0 or edges.max() >= len(nodes):
                raise ValueError("Graph alignment/self edge failure")
            tiles.append((nodes, edges, attrs))
            provenance.append({"alias": alias, "tile": index, "nodes": len(nodes),
                               "edges": edges.shape[1], "node_order_sha256": array_digest(nodes),
                               "edge_sha256": array_digest(edges), "attrs_sha256": array_digest(attrs)})
        if not np.array_equal(np.sort(np.concatenate([t[0] for t in tiles])), np.arange(len(target))):
            raise ValueError("Tile coverage failure")
        cores.append((alias, target, coordinates, tiles))
    if sum(len(c[1]) for c in cores) != 246063:
        raise ValueError("Cell coverage failure")
    archive.write_json("provenance/cohort_and_graph.json", {
        "manifest_sha256": config["dataset"]["dataset_fingerprint"],
        "gene_names": genes, "target_scale": "log1p_raw_counts", "tiles": provenance,
        "target_library_size_available": False, "all_inputs_fit_only": True,
        "known_nondeterminism": ["PyG CUDA scatter reductions"],
    })
    return cores


def verify_pilot(config):
    gate = config["metadata"].get("pilot")
    if not gate:
        raise ValueError("Production requires a verified pilot")
    root = runtime_path(gate["path"])
    if sha256_file(root / "_SUCCESS") != gate["success_sha256"]:
        raise ValueError("Pilot marker changed")
    verify_run_bundle(root, require_success_contract=True)
    diagnostic = json.loads((root / "diagnostics/resource.json").read_text())
    if not diagnostic["pilot_gate_passed"]:
        raise ValueError("Pilot gates failed")
    old = load_yaml_mapping(root / "config.resolved.yaml")
    if old["dataset"] != config["dataset"] or old["metadata"]["implementation_sha256"] != config["metadata"]["implementation_sha256"]:
        raise ValueError("Pilot data or implementation differs")


def fit(model, cores, config, archive, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.0001)
    epochs = config["trainer"]["max_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(epochs):
        start = time.monotonic()
        model.train()
        rng = np.random.default_rng(stable_seed("order", config["seed"], epoch))
        losses, visits, steps, skipped, masked_count = [], 0, 0, 0, 0
        for ci in rng.permutation(len(cores)):
            alias, target, coords, tiles = cores[ci]
            mask = make_mask(coords, 1000, config["masking"]["method"],
                             stable_seed("train-v1", epoch, alias, config["masking"]["method"]))
            total_loss, total_n = 0., 0
            for ti in rng.permutation(len(tiles)):
                nodes, edges, attrs = tiles[ti]
                visits += 1
                local_mask = mask[nodes]
                n = int(local_mask.sum())
                if not n:
                    skipped += 1
                    continue
                x = torch.from_numpy(target[nodes]).to(device)
                m = torch.from_numpy(local_mask).to(device)
                e = torch.from_numpy(edges).to(device)
                a = torch.from_numpy(attrs).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(x, e, edge_attr=a, entry_mask=m)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * n
                total_n += n
                steps += 1
                del x, m, e, a, loss
            if total_n != int(mask.sum()):
                raise ValueError("Training mask coverage mismatch")
            losses.append(total_loss / total_n)
            masked_count += total_n
        scheduler.step()
        row = {"epoch": epoch+1, "mean_core_huber": float(np.mean(losses)),
               "seconds": time.monotonic()-start, "tile_visits": visits,
               "optimizer_steps": steps, "tiles_without_targets": skipped,
               "masked_entries": masked_count, "last_gradient_norm": float(norm)}
        history.append(row)
        archive.append_metric_event({"name": "train/masked_huber", "value": row["mean_core_huber"], "step": epoch+1})
        with (archive.scratch_path / "metrics/epoch_history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno())
        print(json.dumps(row), flush=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite model parameters")
    archive.write_table("metrics/history", history)
    return optimizer, scheduler, history


@torch.no_grad()
def evaluate(model, cores, archive, config, device):
    model.eval()
    rows, identities, canonical = [], [], []
    for alias, target, coords, tiles in cores:
        for method in METHODS:
            for rep in range(3):
                mask = make_mask(coords, 1000, method, stable_seed("evaluation-v1", method, alias, rep))
                identities.append({"alias": alias, "method": method, "replicate": rep,
                                   "sha256": array_digest(mask), "masked_entries": int(mask.sum())})
                baseline = visible_mean(target, mask)
                conditions = ["observed", "zero", "visible_gene_mean", "neighbor_mean"]
                if method == "whole_cell":
                    conditions += ["node_label_permuted", "self_edges_only"]
                accum = {key: ReconstructionMetrics(1000) for key in conditions}
                for ti, (nodes, edges, attrs) in enumerate(tiles):
                    m0, y = mask[nodes], target[nodes]
                    if not m0.any():
                        continue
                    x, m = torch.from_numpy(y).to(device), torch.from_numpy(m0).to(device)
                    a = torch.from_numpy(attrs).to(device)
                    for condition in conditions:
                        if condition == "zero":
                            pred = np.zeros_like(y)
                        elif condition == "visible_gene_mean":
                            pred = np.broadcast_to(baseline, y.shape)
                        elif condition == "neighbor_mean":
                            pred = neighbor_mean(y, m0, edges, baseline)
                        else:
                            edge = edges
                            attr = a
                            if condition == "node_label_permuted":
                                edge = permute_graph_node_labels(edges, len(nodes), stable_seed("null-v1", alias, ti))
                            elif condition == "self_edges_only":
                                edge, attr = edges[:, :0], a[:0]
                            pred = model(x, torch.from_numpy(edge).to(device), edge_attr=attr,
                                         entry_mask=m)[0].cpu().numpy()
                        # Whole-cell masks allow removing entirely unscored rows.
                        selected_rows = m0.any(axis=1)
                        accum[condition].update(y[selected_rows], pred[selected_rows], m0[selected_rows])
                    del x, m, a
                for condition, metrics in accum.items():
                    values = metrics.finalize()
                    if values["n_masked"] != int(mask.sum()):
                        raise ValueError("Evaluation coverage mismatch")
                    rows.append({"train_method": config["masking"]["method"], "model_seed": config["seed"],
                                 "core": alias, "evaluation_method": method, "replicate": rep,
                                 "condition": condition, **values})
                if method == "whole_cell" and rep == 0:
                    b = accum["observed"].base
                    for gene in range(1000):
                        canonical.append({"run_id": archive.run_id,
                            "sample_key": "ag_"+hashlib.sha256(f"{alias}:gene:{gene}:rep0".encode()).hexdigest()[:32],
                            "dataset_id": config["dataset"]["dataset_id"], "split": "fit",
                            "y_true": float(b.gene_sum_true[gene]/b.gene_n[gene]),
                            "y_pred": float(b.gene_sum_prediction[gene]/b.gene_n[gene]),
                            "subgroup_core": alias, "subgroup_gene_index": str(gene)})
        print(f"Evaluated {alias}: all three tasks and fixed masks", flush=True)
    archive.write_json("diagnostics/evaluation_masks.json", identities)
    archive.write_table("metrics/per_core_replicate", rows)
    archive.write_predictions("fit", canonical)
    final = {}
    for method in METHODS:
        conditions = sorted({r["condition"] for r in rows if r["evaluation_method"] == method})
        for condition in conditions:
            selected = [r for r in rows if r["evaluation_method"] == method and r["condition"] == condition]
            fields = [k for k in selected[0] if k not in {"train_method", "model_seed", "core", "evaluation_method", "replicate", "condition"}]
            for field in fields:
                core_values = []
                for alias in ALIASES:
                    values = [r[field] for r in selected if r["core"] == alias and r[field] is not None]
                    if values:
                        core_values.append(float(np.mean(values)))
                final[f"fit/{method}/{condition}/{field}"] = float(np.mean(core_values)) if core_values else None
    final[PRIMARY] = final["fit/whole_cell/observed/masked_huber"]
    archive.write_json("metrics/final.json", final)
    return final


def run(config, archive):
    pilot = validate_protocol(config)
    for name, digest in config["metadata"]["implementation_sha256"].items():
        if sha256_file(runtime_path(name)) != digest:
            raise ValueError(f"Implementation changed: {name}")
    if not pilot:
        verify_pilot(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    start = time.monotonic()
    cores = load_inputs(config, archive)
    io_seconds = time.monotonic()-start
    model = make_source_model(mask_rate=.1).to(device)
    if count_parameters(model) != 6888016:
        raise ValueError("Parameter count drift")
    optimizer, scheduler, history = fit(model, cores, config, archive, device)
    evaluation_start = time.monotonic()
    final = evaluate(model, cores, archive, config, device)
    evaluation_seconds = time.monotonic()-evaluation_start
    payload = {"run_id": archive.run_id, "model_state_dict": {k: v.cpu() for k,v in model.state_dict().items()},
               "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
               "resolved_config": config, "final_epoch": len(history)-1,
               "checkpoint_role": "last", "target_scale": "log1p_raw_counts", "seed": seed}
    buffer = io.BytesIO(); torch.save(payload, buffer)
    checkpoint = archive.write_bytes("checkpoints/last.ckpt", buffer.getvalue())
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded = make_source_model(mask_rate=.1)
    reloaded.load_state_dict(loaded["model_state_dict"], strict=True)
    model.cpu().eval(); reloaded.eval()
    # Deterministic independent CPU replay on a small coordinate-only graph.
    x = torch.from_numpy(cores[0][1][:64].copy())
    coords = cores[0][2][:64]
    edges, attrs = build_symmetric_knn_graph(coords, 15)
    mask = torch.from_numpy(make_mask(coords, 1000, "whole_cell", 7432))
    with torch.no_grad():
        reference = model(x, torch.from_numpy(edges), torch.from_numpy(attrs), mask)[0]
        replay = reloaded(x, torch.from_numpy(edges), torch.from_numpy(attrs), mask)[0]
        hidden_changed = x.clone(); hidden_changed[mask] = 999
        invariant = reloaded(hidden_changed, torch.from_numpy(edges), torch.from_numpy(attrs), mask)[0]
    if not torch.equal(reference, replay) or not torch.equal(replay, invariant):
        raise ValueError("Checkpoint replay or hidden-target invariance failed")
    archive.write_json("diagnostics/replay.json", {"exact_cpu_replay": True, "hidden_target_invariance": True,
                                                 "checkpoint_sha256": sha256_file(checkpoint)})
    peak = torch.cuda.max_memory_allocated()/1024**3
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    projected_hours = float((io_seconds + np.mean([r["seconds"] for r in history])*200 + evaluation_seconds)/3600)
    free = shutil.disk_usage(archive.scratch_path).free/1024**3
    gates = resource_gates(peak, host, projected_hours, free,
                          all(r["tile_visits"] == sum(len(c[3]) for c in cores) for r in history))
    resources = {"gpu_name": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "torch": str(torch.__version__), "cuda": torch.version.cuda, "precision": "fp32",
                 "peak_allocated_vram_gib": peak, "peak_reserved_vram_gib": torch.cuda.max_memory_reserved()/1024**3,
                 "peak_host_memory_gib": host, "total_seconds": time.monotonic()-start,
                 "io_and_graph_seconds": io_seconds, "evaluation_seconds": evaluation_seconds,
                 "projected_200_epoch_hours": projected_hours, "free_disk_gib": free,
                 "checks": gates, "pilot_gate_passed": all(gates.values())}
    archive.write_json("diagnostics/resource.json", resources)
    if pilot and not all(gates.values()):
        raise RuntimeError("Resource pilot gate failed")
    archive.write_summary({"run_id": archive.run_id, "status": "success",
        "primary_metric_name": PRIMARY, "primary_metric_value": final[PRIMARY],
        "completed_global_epochs": len(history), "final_epoch": len(history)-1,
        "checkpoint_role": "last", "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_count": 6888016, "model_seed": seed,
        "train_method": config["masking"]["method"], "mask_rate": .1,
        "prediction_granularity": "per_core_gene_masked_means_not_individual_cells",
        "generalization_claim_supported": False, "resources": resources})
    print(json.dumps({"complete": True, "primary_huber": final[PRIMARY], "resources": resources}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-scratch", type=Path, required=True)
    args = parser.parse_args()
    run_id = os.environ["BAGM_RUN_ID"]
    if args.run_scratch.resolve() != Path(os.environ["BAGM_RUN_SCRATCH"]).resolve():
        raise ValueError("Worker scratch mismatch")
    if args.config.resolve() != (args.run_scratch / "config.resolved.yaml").resolve():
        raise ValueError("Worker config mismatch")
    run(load_yaml_mapping(args.config), RunArchive.attach_active(run_id, paths=current_paths(), scratch_path=args.run_scratch))

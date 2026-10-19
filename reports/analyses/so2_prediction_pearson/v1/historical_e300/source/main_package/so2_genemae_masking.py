"""Matched SO_2 GeneMAE masking tasks; no target-derived input scaling."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix

from .myjju_genemae import MaskedRegressionAccumulator
from .paths import current_paths

CAMPAIGN = "cmp_20261005_so2_genemae_masking"
PROTOCOL = "held_in_so2_genemae_masking_v1"
METHODS = ("whole_cell", "partial_gene", "spatial_block")
ALIASES = tuple(f"SO2-C{i}" for i in range(15, 29))
PRIMARY = "fit/whole_node/masked_huber"


def resource_gates(peak_gib, host_gib, projected_hours, free_gib, coverage):
    """Use Python booleans, including when resource estimates are NumPy scalars."""
    return {"vram": bool(peak_gib <= 21), "host": bool(host_gib <= 40),
            "runtime": bool(projected_hours <= 6), "disk": bool(free_gib-3 >= 8),
            "tile_coverage": bool(coverage)}


def runtime_path(reference: str) -> Path:
    p = current_paths()
    value = Path(reference)
    if value.is_absolute():
        return value
    if ".." in value.parts:
        raise ValueError("Unsafe path")
    roots = {"data": p.data_root, "artifacts": p.artifact_root,
             "scratch": p.scratch_root, "state": p.state_root,
             "reports": p.report_root, "results": p.result_root}
    if value.parts[0] in roots:
        return roots[value.parts[0]].joinpath(*value.parts[1:])
    return p.project_root / value


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "big") % (2**63-1)


def array_digest(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def make_mask(coordinates: np.ndarray, genes: int, method: str, seed: int,
              rate: float = 0.1) -> np.ndarray:
    """Exact core-level cell count or per-cell gene count, sampled without replacement."""
    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Coordinates must be finite [cells,2]")
    if len(coords) < 1 or genes < 1 or not 0 < rate < 1 or method not in METHODS:
        raise ValueError("Invalid mask dimensions, rate or method")
    rng = np.random.default_rng(seed)
    mask = np.zeros((len(coords), genes), dtype=bool)
    if method == "partial_gene":
        k = max(1, int(math.floor(genes * rate + 0.5)))
        # Chunk temporaries, keeping the selection distribution identical per row.
        for start in range(0, len(coords), 2048):
            stop = min(start + 2048, len(coords))
            values = rng.random((stop-start, genes), dtype=np.float32)
            chosen = np.argpartition(values, k-1, axis=1)[:, :k]
            mask[np.arange(start, stop)[:, None], chosen] = True
    else:
        k = max(1, int(math.floor(len(coords) * rate + 0.5)))
        if method == "whole_cell":
            chosen = rng.choice(len(coords), size=k, replace=False)
        else:
            anchor = int(rng.integers(len(coords)))
            distances = np.square(coords - coords[anchor]).sum(axis=1)
            chosen = np.argsort(distances, kind="stable")[:k]
        mask[chosen, :] = True
    return mask


def visible_mean(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = (~mask).sum(axis=0)
    return np.divide(np.where(mask, 0, target).sum(axis=0, dtype=np.float64),
                     count, out=np.zeros(target.shape[1], dtype=np.float64),
                     where=count > 0).astype(np.float32)


def neighbor_mean(target: np.ndarray, mask: np.ndarray, edges: np.ndarray,
                  fallback: np.ndarray) -> np.ndarray:
    n = len(target)
    matrix = csr_matrix((np.ones(edges.shape[1], dtype=np.float32),
                         (edges[1], edges[0])), shape=(n, n))
    sums = matrix @ np.where(mask, 0, target)
    counts = matrix @ (~mask).astype(np.float32)
    return np.divide(sums, counts, out=np.broadcast_to(fallback, target.shape).copy(),
                     where=counts > 0)


class ReconstructionMetrics:
    """Existing validated regression metrics plus sparsity-stratified errors."""
    def __init__(self, genes: int):
        self.base = MaskedRegressionAccumulator(genes)
        self.parts = {name: np.zeros(4, dtype=np.float64) for name in ("positive", "zero")}

    def update(self, target, prediction, mask):
        self.base.update(target, prediction, mask)
        y, p = target[mask], prediction[mask]
        for name, selected in (("positive", y > 0), ("zero", y == 0)):
            d = np.abs(p[selected].astype(np.float64) - y[selected])
            self.parts[name] += [len(d), np.minimum(d, 1).__pow__(2).sum() * .5
                                + np.maximum(d-1, 0).sum(), (d*d).sum(), d.sum()]

    def finalize(self):
        result = self.base.finalize()
        for name, (n, huber, mse, mae) in self.parts.items():
            result[f"{name}_count"] = int(n)
            for metric, value in (("huber", huber), ("mse", mse), ("mae", mae)):
                result[f"{name}_{metric}"] = float(value/n) if n else None
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in result.items()}


def validate_protocol(config):
    from .configuration import validate_experiment_config
    validate_experiment_config(config)
    expected = {
        ("campaign", "campaign_id"): CAMPAIGN,
        ("model", "name"): "myjju-genemae",
        ("model", "expected_trainable_parameters"): 6888016,
        ("dataset", "target_scale"): "log1p_raw_counts",
        ("features", "node_features"): ["visible_expression", "mask_token"],
        ("graph", "neighbor_k"): 15,
        ("graph", "maximum_tile_nodes"): 7000,
        ("masking", "rate"): .1,
        ("trainer", "learning_rate"): .0015,
        ("trainer", "weight_decay"): .0001,
        ("trainer", "precision"): "fp32",
        ("trainer", "gradient_clip_norm"): 5.,
        ("evaluation", "protocol"): PROTOCOL,
        ("evaluation", "primary_metric"): PRIMARY,
        ("evaluation", "methods"): list(METHODS),
        ("evaluation", "replicates"): 3,
    }
    for (section, field), value in expected.items():
        if config[section].get(field) != value:
            raise ValueError(f"Protocol drift: {section}.{field}")
    pilot = config["metadata"]["execution_role"] == "resource_pilot"
    if config["trainer"]["max_epochs"] != (2 if pilot else 200):
        raise ValueError("Epoch budget drift")
    if config["seed"] not in (0, 1, 2) or config["masking"]["method"] not in METHODS:
        raise ValueError("Unexpected seed or masking method")
    return pilot

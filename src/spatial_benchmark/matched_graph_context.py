"""Matched graph-context model and leakage-safe preprocessing primitives.

This module deliberately keeps graph construction outside the learned model.  Each
arm supplies a 1,000-dimensional context tensor to the *same* additive network.
Consequently an arm can change values, but cannot silently change architecture or
trainable parameter count.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .identifiers import canonical_json, canonical_sha256


CAMPAIGN_ID = "cmp_20260812_matched_graph_context_nested_cv"
CONTRACT_SHA256 = "4e39e1ef623a5ce4e1d67f4733bdea790000745d523a8845bcd11a0ddea77b73"
PREPARED_MANIFEST_SHA256 = "e59546beeed2e8e2523248e89ac6738bf90aa65f28248a42a7091c54255d0f32"
INTEGRITY_MANIFEST_SHA256 = "65a6dde8e9aa7f733bb1f06075c3a0dedd2084a89c55eaa575c91276a45aff18"
PROCESSED_FINGERPRINT = "01c525695883784befc1b9ebbe37a6d96b248d5b18450f46a1255e571bd3819e"
SPLIT_FINGERPRINT = "12c0d46244ed443a482586fc85422672f9f04132c7def49a741c40ba48bf4264"
DATASET_ID = "gastric_cosmx_drive_public"
DATASET_VERSION = "drive_snapshot_20260810"
SPLIT_ID = "opaque_geometry_components_075mm_4fold_v1"

ARMS = ("no_graph", "observed_near", "permuted_near", "observed_annular")
GENE_COUNT = 1000
MORPHOLOGY_COUNT = 22
BASE_FEATURE_COUNT = 27
PROJECTION_SEED = 2026081201
EVALUATION_EPOCHS = (12, 24, 48, 96, 192)
CONFIRMATION_SEEDS = (20260812, 20261812, 20262812, 20263812, 20264812)
TUNING_SEEDS = (20260812, 20261812, 20262812)
FAITHFULNESS_CONTEXTS = (
    "native",
    "zero",
    "permuted_near",
    "observed_annular",
    "observed_near_10_25",
)

_CANDIDATE_ROWS = (
    ("c00", 32, 0.0003, 0.0, 0.0),
    ("c01", 32, 0.0003, 0.001, 0.2),
    ("c02", 32, 0.001, 0.0001, 0.1),
    ("c03", 32, 0.003, 0.0, 0.2),
    ("c04", 32, 0.003, 0.001, 0.0),
    ("c05", 64, 0.0003, 0.0, 0.1),
    ("c06", 64, 0.0003, 0.001, 0.0),
    ("c07", 64, 0.001, 0.0, 0.2),
    ("c08", 64, 0.001, 0.0001, 0.1),
    ("c09", 64, 0.003, 0.0001, 0.0),
    ("c10", 64, 0.003, 0.001, 0.2),
    ("c11", 128, 0.0003, 0.0001, 0.2),
    ("c12", 128, 0.0003, 0.001, 0.0),
    ("c13", 128, 0.001, 0.0, 0.0),
    ("c14", 128, 0.001, 0.001, 0.1),
    ("c15", 128, 0.003, 0.0001, 0.1),
)
CANDIDATES: Mapping[str, Mapping[str, Any]] = {
    row[0]: {
        "candidate_id": row[0],
        "hidden_width": row[1],
        "learning_rate": row[2],
        "weight_decay": row[3],
        "dropout": row[4],
    }
    for row in _CANDIDATE_ROWS
}


class MatchedGraphContextError(RuntimeError):
    """Raised when execution would depart from the frozen comparison."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def candidate_config(candidate_id: str) -> dict[str, Any]:
    """Return a defensive copy of one frozen search candidate."""

    try:
        return dict(CANDIDATES[candidate_id])
    except KeyError as error:
        raise MatchedGraphContextError(
            f"Unknown frozen candidate {candidate_id!r}; expected c00 through c15."
        ) from error


class MatchedAdditiveContextMLP(nn.Module):
    """The literal parameter-matched architecture used by every learned arm."""

    def __init__(
        self,
        *,
        hidden_width: int,
        dropout: float,
        base_features: int = BASE_FEATURE_COUNT,
        genes: int = GENE_COUNT,
    ) -> None:
        super().__init__()
        if hidden_width < 1 or genes < 1 or base_features < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.base_features = int(base_features)
        self.genes = int(genes)
        self.hidden_width = int(hidden_width)
        self.dropout_probability = float(dropout)
        self.base_linear = nn.Linear(self.base_features, self.genes)
        self.context_linear = nn.Linear(self.genes, self.genes, bias=False)
        self.context_hidden = nn.Linear(self.genes, self.hidden_width)
        self.activation = nn.GELU(approximate="none")
        self.dropout = nn.Dropout(self.dropout_probability)
        self.context_output = nn.Linear(self.hidden_width, self.genes, bias=False)

    def forward(self, base: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if base.ndim != 2 or base.shape[1] != self.base_features:
            raise ValueError("base tensor has the wrong shape")
        if context.ndim != 2 or context.shape[1] != self.genes:
            raise ValueError("context tensor has the wrong shape")
        hidden = self.dropout(self.activation(self.context_hidden(context)))
        return (
            self.base_linear(base)
            + self.context_linear(context)
            + self.context_output(hidden)
        )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_matched_model(
    arm: str,
    *,
    hidden_width: int,
    dropout: float,
    seed: int,
    genes: int = GENE_COUNT,
) -> MatchedAdditiveContextMLP:
    """Build an arm while making it impossible for the arm to alter parameters."""

    if arm not in ARMS:
        raise MatchedGraphContextError(f"Unknown arm: {arm!r}")
    torch.manual_seed(int(seed))
    return MatchedAdditiveContextMLP(
        hidden_width=hidden_width,
        dropout=dropout,
        genes=genes,
    )


def frozen_rank27_projection(
    *, seed: int = PROJECTION_SEED, genes: int = GENE_COUNT
) -> np.ndarray:
    """Create the deterministic, full-row-rank no-graph projection.

    QR on a 1000-by-27 Gaussian draw gives 27 orthonormal rows after transpose,
    making the rank invariant explicit instead of relying on a likely-full-rank draw.
    """

    if genes < BASE_FEATURE_COUNT:
        raise ValueError("projection output must be at least rank 27")
    generator = np.random.default_rng(int(seed))
    draw = generator.standard_normal((genes, BASE_FEATURE_COUNT))
    q, _ = np.linalg.qr(draw, mode="reduced")
    projection = np.ascontiguousarray(q.T, dtype=np.float32)
    if np.linalg.matrix_rank(projection.astype(np.float64)) != BASE_FEATURE_COUNT:
        raise MatchedGraphContextError("frozen no-graph projection lost rank")
    return projection


@dataclass(frozen=True)
class ScaleState:
    mean: np.ndarray
    scale: np.ndarray
    count: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise MatchedGraphContextError("normalization produced a nonfinite value")
        return np.asarray(result, dtype=np.float32)

    def payload(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_sha256": ndarray_sha256(self.mean),
            "scale_sha256": ndarray_sha256(self.scale),
            "dimensions": int(self.mean.size),
        }


class StreamingMoments:
    """Stable-enough float64 population moments for chunked training-only fits."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = int(dimensions)
        self.count = 0
        self.sum = np.zeros(self.dimensions, dtype=np.float64)
        self.sum_square = np.zeros(self.dimensions, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != self.dimensions:
            raise ValueError("moment input has the wrong shape")
        if not np.isfinite(array).all():
            raise MatchedGraphContextError("normalization fit encountered nonfinite input")
        self.count += int(array.shape[0])
        self.sum += array.sum(axis=0, dtype=np.float64)
        self.sum_square += np.square(array, dtype=np.float64).sum(axis=0, dtype=np.float64)

    def finish(self, *, floor: float = 1.0e-6) -> ScaleState:
        if self.count < 1:
            raise MatchedGraphContextError("cannot fit normalization on an empty training split")
        mean = self.sum / self.count
        variance = np.maximum(self.sum_square / self.count - mean * mean, 0.0)
        scale = np.maximum(np.sqrt(variance), float(floor))
        return ScaleState(
            np.asarray(mean, dtype=np.float32),
            np.asarray(scale, dtype=np.float32),
            self.count,
        )


@dataclass(frozen=True)
class PreprocessingState:
    coordinate: Mapping[str, ScaleState]
    base: ScaleState
    target: ScaleState
    context: ScaleState
    projection_sha256: str
    training_folds: tuple[int, ...]

    def payload(self) -> dict[str, Any]:
        value = {
            "fit_scope": "training_only",
            "training_folds": list(self.training_folds),
            "coordinate_by_slide": {
                slide: state.payload() for slide, state in sorted(self.coordinate.items())
            },
            "base": self.base.payload(),
            "target": self.target.payload(),
            "context": self.context.payload(),
            "projection_sha256": self.projection_sha256,
        }
        value["state_sha256"] = canonical_sha256(value)
        return value


def broad_field_basis(coordinates: np.ndarray, state: ScaleState) -> np.ndarray:
    z = state.transform(coordinates)
    if z.shape[1] != 2:
        raise ValueError("coordinates must have x and y columns")
    x, y = z[:, 0], z[:, 1]
    return np.column_stack((x, y, x * x, x * y, y * y)).astype(np.float32)


def raw_base_features(
    metadata: np.ndarray,
    coordinates: np.ndarray,
    coordinate_state: ScaleState,
) -> np.ndarray:
    morphology = np.asarray(metadata, dtype=np.float32)
    if morphology.ndim != 2 or morphology.shape[1] != MORPHOLOGY_COUNT:
        raise ValueError("metadata must contain exactly 22 permitted fields")
    result = np.concatenate(
        (morphology, broad_field_basis(coordinates, coordinate_state)), axis=1
    )
    if result.shape[1] != BASE_FEATURE_COUNT or not np.isfinite(result).all():
        raise MatchedGraphContextError("base features violate the frozen 27-field contract")
    return np.asarray(result, dtype=np.float32)


def no_graph_context(base_normalized: np.ndarray, projection: np.ndarray) -> np.ndarray:
    if base_normalized.ndim != 2 or base_normalized.shape[1] != BASE_FEATURE_COUNT:
        raise ValueError("normalized base input must have 27 columns")
    if projection.shape[0] != BASE_FEATURE_COUNT:
        raise ValueError("projection must have 27 rows")
    return np.asarray(base_normalized @ projection, dtype=np.float32)


def split_roles(mode: str, outer_fold: int) -> dict[str, Any]:
    if outer_fold not in range(4):
        raise MatchedGraphContextError("fold must be 0 through 3")
    if mode == "tune":
        validation = (outer_fold + 1) % 4
        train = tuple(fold for fold in range(4) if fold not in {outer_fold, validation})
        return {
            "train_folds": train,
            "validation_fold": validation,
            "excluded_fold": outer_fold,
        }
    if mode == "confirm":
        return {
            "train_folds": tuple(fold for fold in range(4) if fold != outer_fold),
            "test_fold": outer_fold,
        }
    raise MatchedGraphContextError(f"unknown real-data mode {mode!r}")


def validate_component_disjointness(
    folds: np.ndarray, components: np.ndarray, eligible: np.ndarray
) -> None:
    selected = np.flatnonzero(np.asarray(eligible, dtype=bool))
    seen: dict[int, int] = {}
    for component, fold in zip(components[selected], folds[selected], strict=True):
        component_int, fold_int = int(component), int(fold)
        previous = seen.setdefault(component_int, fold_int)
        if previous != fold_int:
            raise MatchedGraphContextError(
                f"geometry component {component_int} crosses folds {previous} and {fold_int}"
            )


def validate_prepared_root(root: str | Path) -> dict[str, Any]:
    """Fail closed on the checksum-bound V0 authority files."""

    prepared = Path(root)
    manifest_path = prepared / "manifest.json"
    integrity_path = prepared / "integrity_manifest.json"
    if sha256_file(manifest_path) != PREPARED_MANIFEST_SHA256:
        raise MatchedGraphContextError("prepared manifest checksum changed")
    if sha256_file(integrity_path) != INTEGRITY_MANIFEST_SHA256:
        raise MatchedGraphContextError("integrity manifest checksum changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("processed_fingerprint") != PROCESSED_FINGERPRINT:
        raise MatchedGraphContextError("processed fingerprint changed")
    if manifest.get("split_fingerprint") != SPLIT_FINGERPRINT:
        raise MatchedGraphContextError("split fingerprint changed")
    return manifest


def selection_payload_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in {"payload_sha256", "receipt_payload_sha256", "checksum"}
        }
    )


def validate_selection_receipt(
    path: str | Path, *, arm: str, outer_fold: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate frozen all-arm selection and return this arm's configuration."""

    receipt_path = Path(path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MatchedGraphContextError("selection receipt must be a JSON mapping")
    required = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "contract_sha256": CONTRACT_SHA256,
        "status": "frozen",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise MatchedGraphContextError(f"selection receipt {key} is not frozen authority")
    if isinstance(outer_fold, bool) or outer_fold not in range(4):
        raise MatchedGraphContextError("confirmation outer_fold must be 0 through 3")
    if payload.get("kind", payload.get("receipt_kind")) != "matched_graph_context_selection_receipt":
        raise MatchedGraphContextError("selection receipt kind is not recognized")
    if payload.get("test_metrics_used_for_selection") is not False:
        raise MatchedGraphContextError("selection receipt is not validation-only authority")
    if payload.get("cross_outer_pooling") is not False:
        raise MatchedGraphContextError("selection receipt does not prohibit cross-outer pooling")
    calculated = selection_payload_sha256(payload)
    digest = payload.get("payload_sha256")
    if digest != calculated:
        raise MatchedGraphContextError("selection receipt payload checksum does not verify")
    sources_by_fold = payload.get("source_tuning_result_sha256s_by_outer_fold")
    if (
        not isinstance(sources_by_fold, Mapping)
        or set(sources_by_fold) != {str(fold) for fold in range(4)}
    ):
        raise MatchedGraphContextError(
            "selection receipt lacks outer-fold-specific tuning result bindings"
        )
    for fold in range(4):
        sources = sources_by_fold[str(fold)]
        if (
            not isinstance(sources, list)
            or len(sources) != 88
            or len(set(sources)) != 88
            or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in sources
            )
        ):
            raise MatchedGraphContextError(
                f"selection receipt must bind 88 unique tuning results for outer fold {fold}"
            )
    selections_by_fold = payload.get("selected_by_outer_fold")
    if (
        not isinstance(selections_by_fold, Mapping)
        or set(selections_by_fold) != {str(fold) for fold in range(4)}
    ):
        raise MatchedGraphContextError(
            "selection receipt must contain four independent outer-fold selections"
        )
    selections = selections_by_fold[str(outer_fold)]
    if not isinstance(selections, Mapping) or set(selections) != set(ARMS):
        raise MatchedGraphContextError(
            "selection for this outer fold must select all four arms exactly"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for selected_arm in ARMS:
        block = selections[selected_arm]
        if not isinstance(block, Mapping):
            raise MatchedGraphContextError(f"selection for {selected_arm} is malformed")
        raw = block.get("config", block.get("hyperparameters"))
        if not isinstance(raw, Mapping):
            raise MatchedGraphContextError(
                f"selection for {selected_arm} lacks a configuration mapping"
            )
        candidate_id = str(block.get("candidate_id", raw.get("candidate_id", "")))
        expected = candidate_config(candidate_id)
        hidden = raw.get("hidden_width", raw.get("hidden"))
        actual = {
            "candidate_id": candidate_id,
            "hidden_width": int(hidden) if not isinstance(hidden, bool) else -1,
            "learning_rate": float(raw.get("learning_rate", math.nan)),
            "weight_decay": float(raw.get("weight_decay", math.nan)),
            "dropout": float(raw.get("dropout", math.nan)),
        }
        if actual != expected:
            raise MatchedGraphContextError(
                f"selection for {selected_arm} does not match frozen {candidate_id}"
            )
        epoch = raw.get("epoch", raw.get("selected_epoch"))
        if isinstance(epoch, bool) or epoch not in EVALUATION_EPOCHS:
            raise MatchedGraphContextError(f"selection for {selected_arm} has invalid epoch")
        actual["epoch"] = int(epoch)
        if raw.get("batch_size") != 4096:
            raise MatchedGraphContextError(
                f"selection for {selected_arm} does not retain batch_size=4096"
            )
        actual["batch_size"] = 4096
        expected_config_sha256 = canonical_sha256(actual)
        if block.get("config_sha256") != expected_config_sha256:
            raise MatchedGraphContextError(
                f"selection config checksum does not verify for {selected_arm}"
            )
        parameter_count = block.get("parameter_count", raw.get("parameter_count"))
        expected_parameter_count = trainable_parameter_count(
            build_matched_model(
                selected_arm,
                hidden_width=actual["hidden_width"],
                dropout=actual["dropout"],
                seed=0,
            )
        )
        if parameter_count != expected_parameter_count:
            raise MatchedGraphContextError(
                f"selection parameter count does not verify for {selected_arm}"
            )
        normalized[selected_arm] = actual
    widths = {config["hidden_width"] for config in normalized.values()}
    if len(widths) != 1:
        raise MatchedGraphContextError(
            "confirmation selection violates the shared-hidden parameter match"
        )
    return payload, normalized[arm]


class MetricAccumulator:
    """Streaming component and gene regression metrics without retaining cells."""

    def __init__(self, genes: int = GENE_COUNT) -> None:
        self.genes = int(genes)
        self.components: dict[tuple[str, int], list[float]] = {}
        self.component_genes: dict[
            tuple[str, int], tuple[np.ndarray, np.ndarray, int]
        ] = {}
        self.n = 0
        self.sum_true = np.zeros(genes, dtype=np.float64)
        self.sum_pred = np.zeros(genes, dtype=np.float64)
        self.sum_true_square = np.zeros(genes, dtype=np.float64)
        self.sum_pred_square = np.zeros(genes, dtype=np.float64)
        self.sum_cross = np.zeros(genes, dtype=np.float64)
        self.sum_square_error = np.zeros(genes, dtype=np.float64)
        self.sum_absolute_error = np.zeros(genes, dtype=np.float64)

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        slide: str,
        components: np.ndarray,
    ) -> None:
        true = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
        component_values = np.asarray(components)
        if true.shape != pred.shape or true.ndim != 2 or true.shape[1] != self.genes:
            raise ValueError("metric tensors have incompatible shapes")
        if component_values.shape != (true.shape[0],):
            raise ValueError("component vector has incompatible shape")
        if not np.isfinite(true).all() or not np.isfinite(pred).all():
            raise MatchedGraphContextError("evaluation encountered nonfinite predictions")
        error = pred - true
        square = error * error
        absolute = np.abs(error)
        for component in np.unique(component_values):
            mask = component_values == component
            key = (slide, int(component))
            current = self.components.setdefault(key, [0.0, 0.0, 0.0])
            current[0] += float(square[mask].sum(dtype=np.float64))
            current[1] += float(absolute[mask].sum(dtype=np.float64))
            current[2] += int(mask.sum())
            gene_sse, gene_sae, gene_n = self.component_genes.get(
                key,
                (
                    np.zeros(self.genes, dtype=np.float64),
                    np.zeros(self.genes, dtype=np.float64),
                    0,
                ),
            )
            gene_sse += square[mask].sum(axis=0, dtype=np.float64)
            gene_sae += absolute[mask].sum(axis=0, dtype=np.float64)
            self.component_genes[key] = (gene_sse, gene_sae, gene_n + int(mask.sum()))
        self.n += true.shape[0]
        self.sum_true += true.sum(axis=0)
        self.sum_pred += pred.sum(axis=0)
        self.sum_true_square += (true * true).sum(axis=0)
        self.sum_pred_square += (pred * pred).sum(axis=0)
        self.sum_cross += (true * pred).sum(axis=0)
        self.sum_square_error += square.sum(axis=0)
        self.sum_absolute_error += absolute.sum(axis=0)

    def component_rows(self, **fields: Any) -> list[dict[str, Any]]:
        rows = []
        for (slide, component), (sse, sae, n_cells_float) in sorted(self.components.items()):
            n_cells = int(n_cells_float)
            denominator = n_cells * self.genes
            rows.append(
                {
                    **fields,
                    "slide": slide,
                    "component": component,
                    "n_cells": n_cells,
                    "mse": sse / denominator,
                    "mae": sae / denominator,
                }
            )
        return rows

    def gene_rows(self, genes: Sequence[str], **fields: Any) -> list[dict[str, Any]]:
        if self.n < 1 or len(genes) != self.genes or len(set(genes)) != self.genes:
            raise MatchedGraphContextError("gene metric coverage is incomplete")
        numerator = self.sum_cross - self.sum_true * self.sum_pred / self.n
        true_ss = np.maximum(self.sum_true_square - self.sum_true**2 / self.n, 0.0)
        pred_ss = np.maximum(self.sum_pred_square - self.sum_pred**2 / self.n, 0.0)
        denominator = np.sqrt(true_ss * pred_ss)
        pearson = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        return [
            {
                **fields,
                "gene_index": index,
                "gene": str(gene),
                "n_cells": self.n,
                "mse": float(self.sum_square_error[index] / self.n),
                "mae": float(self.sum_absolute_error[index] / self.n),
                "pearson": float(pearson[index]),
            }
            for index, gene in enumerate(genes)
        ]

    def component_gene_rows(
        self, genes: Sequence[str], **fields: Any
    ) -> list[dict[str, Any]]:
        if len(genes) != self.genes or len(set(genes)) != self.genes:
            raise MatchedGraphContextError("component-gene axis is incomplete")
        rows: list[dict[str, Any]] = []
        for (slide, component), (sse, sae, n_cells) in sorted(
            self.component_genes.items()
        ):
            if n_cells < 1:
                raise MatchedGraphContextError("component-gene coverage is empty")
            rows.extend(
                {
                    **fields,
                    "slide": slide,
                    "component": component,
                    "n_cells": n_cells,
                    "gene_index": index,
                    "gene": str(gene),
                    "mse": float(sse[index] / n_cells),
                    "mae": float(sae[index] / n_cells),
                }
                for index, gene in enumerate(genes)
            )
        return rows

    def component_equal(self) -> tuple[float, float]:
        rows = self.component_rows()
        if not rows:
            raise MatchedGraphContextError("no evaluation components were accumulated")
        return (
            float(np.mean([row["mse"] for row in rows])),
            float(np.mean([row["mae"] for row in rows])),
        )


def short_edge_removed_context(
    *,
    receiver_indices: np.ndarray,
    coordinates: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    source_expression: np.ndarray,
    minimum_um: float = 10.0,
    maximum_um: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute means after filtering only endpoints in the frozen near CSR.

    No new edge can be introduced.  A receiver with no remaining source obtains
    a literal raw all-zero vector and degree zero, as required by the contract.
    """

    receivers = np.asarray(receiver_indices, dtype=np.int64)
    result = np.zeros((receivers.size, source_expression.shape[1]), dtype=np.float32)
    degree = np.zeros(receivers.size, dtype=np.int16)
    lower_square, upper_square = minimum_um**2, maximum_um**2
    for output_index, receiver in enumerate(receivers):
        start, stop = int(indptr[receiver]), int(indptr[receiver + 1])
        sources = np.asarray(indices[start:stop], dtype=np.int64)
        if sources.size == 0:
            continue
        delta = np.asarray(coordinates[sources], dtype=np.float64) - np.asarray(
            coordinates[receiver], dtype=np.float64
        )
        distance_square = np.einsum("ij,ij->i", delta, delta)
        kept = sources[(distance_square >= lower_square) & (distance_square <= upper_square)]
        if kept.size:
            result[output_index] = np.asarray(
                source_expression[kept], dtype=np.float32
            ).mean(axis=0, dtype=np.float32)
            degree[output_index] = kept.size
    if not np.isfinite(result).all():
        raise MatchedGraphContextError("10-25 um context contains nonfinite values")
    return result, degree


def _ridge_validation_mse(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    ridge: float = 1.0e-3,
) -> float:
    gram = train_x.T @ train_x + ridge * np.eye(train_x.shape[1])
    weights = np.linalg.solve(gram, train_x.T @ train_y)
    error = validation_x @ weights - validation_y
    return float(np.mean(error * error))


def deterministic_synthetic_gate(
    *,
    seed: int = 20260812,
    planted_context: bool,
    samples: int = 3072,
) -> dict[str, Any]:
    """Run a fast matched-dimension positive or null recovery control.

    The gate uses deterministic ridge fits to isolate whether a correctly aligned
    context contains held-out information.  Both compared designs have identical
    column count; the control differs only by observed versus frozen-projected
    context values.  It is intentionally independent of stochastic GPU kernels.
    """

    rng = np.random.default_rng(seed)
    base_dim, context_dim, targets = 8, 12, 6
    base = rng.normal(size=(samples, base_dim))
    context = rng.normal(size=(samples, context_dim))
    projection = rng.normal(size=(base_dim, context_dim)) / math.sqrt(base_dim)
    no_graph = base @ projection
    base_effect = base @ rng.normal(size=(base_dim, targets))
    context_effect = context @ rng.normal(size=(context_dim, targets))
    noise = rng.normal(scale=0.35, size=(samples, targets))
    target = base_effect + noise
    if planted_context:
        target = target + 1.25 * context_effect
    permutation = rng.permutation(samples)
    split = samples * 2 // 3
    train, validation = permutation[:split], permutation[split:]

    def design(context_values: np.ndarray) -> np.ndarray:
        return np.concatenate((base, context_values), axis=1)

    observed_mse = _ridge_validation_mse(
        design(context)[train], target[train], design(context)[validation], target[validation]
    )
    control_mse = _ridge_validation_mse(
        design(no_graph)[train], target[train], design(no_graph)[validation], target[validation]
    )
    relative_gain = (control_mse - observed_mse) / control_mse
    threshold = 0.05 if planted_context else 0.01
    passed = relative_gain >= threshold if planted_context else abs(relative_gain) <= threshold
    result = {
        "schema_version": 1,
        "kind": "positive" if planted_context else "null",
        "seed": seed,
        "observed_mse": observed_mse,
        "no_graph_mse": control_mse,
        "relative_gain": relative_gain,
        "threshold": threshold,
        "passed": bool(passed),
    }
    if not all(math.isfinite(float(result[key])) for key in ("observed_mse", "no_graph_mse", "relative_gain")):
        raise MatchedGraphContextError("synthetic gate produced a nonfinite metric")
    return result


def run_synthetic_recovery_gates(seed: int = 20260812) -> dict[str, Any]:
    positive = deterministic_synthetic_gate(seed=seed, planted_context=True)
    null = deterministic_synthetic_gate(seed=seed, planted_context=False)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "positive": positive,
        "null": null,
        "passed": bool(positive["passed"] and null["passed"]),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "ARMS",
    "BASE_FEATURE_COUNT",
    "CAMPAIGN_ID",
    "CANDIDATES",
    "CONFIRMATION_SEEDS",
    "CONTRACT_SHA256",
    "DATASET_ID",
    "DATASET_VERSION",
    "EVALUATION_EPOCHS",
    "FAITHFULNESS_CONTEXTS",
    "GENE_COUNT",
    "INTEGRITY_MANIFEST_SHA256",
    "MatchedAdditiveContextMLP",
    "MatchedGraphContextError",
    "MetricAccumulator",
    "PREPARED_MANIFEST_SHA256",
    "PROCESSED_FINGERPRINT",
    "PROJECTION_SEED",
    "PreprocessingState",
    "SPLIT_FINGERPRINT",
    "SPLIT_ID",
    "ScaleState",
    "StreamingMoments",
    "TUNING_SEEDS",
    "broad_field_basis",
    "build_matched_model",
    "candidate_config",
    "deterministic_synthetic_gate",
    "frozen_rank27_projection",
    "ndarray_sha256",
    "no_graph_context",
    "raw_base_features",
    "run_synthetic_recovery_gates",
    "selection_payload_sha256",
    "sha256_file",
    "short_edge_removed_context",
    "split_roles",
    "trainable_parameter_count",
    "validate_component_disjointness",
    "validate_prepared_root",
    "validate_selection_receipt",
]

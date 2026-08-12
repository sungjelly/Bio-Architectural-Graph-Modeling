#!/usr/bin/env python3
"""Run one matched graph-context tuning or confirmation job.

Tuning jobs are deliberately validation-only and publish neither a checkpoint
nor any outer-evaluation output.  Confirmation requires a checksum-bound frozen
all-arm selection receipt, refits on every non-test fold, and publishes the
locked checkpoint plus component/gene and context-substitution diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.identifiers import (  # noqa: E402
    canonical_json,
    canonical_sha256,
    scientific_id,
)
from spatial_benchmark.matched_graph_context import (  # noqa: E402
    ARMS,
    BASE_FEATURE_COUNT,
    CAMPAIGN_ID,
    CONFIRMATION_SEEDS,
    CONTRACT_SHA256,
    DATASET_ID,
    DATASET_VERSION,
    EVALUATION_EPOCHS,
    FAITHFULNESS_CONTEXTS,
    GENE_COUNT,
    INTEGRITY_MANIFEST_SHA256,
    MatchedAdditiveContextMLP,
    MatchedGraphContextError,
    MetricAccumulator,
    PREPARED_MANIFEST_SHA256,
    PROCESSED_FINGERPRINT,
    PreprocessingState,
    PROJECTION_SEED,
    SPLIT_FINGERPRINT,
    SPLIT_ID,
    ScaleState,
    StreamingMoments,
    TUNING_SEEDS,
    build_matched_model,
    candidate_config,
    frozen_rank27_projection,
    ndarray_sha256,
    no_graph_context,
    raw_base_features,
    run_synthetic_recovery_gates,
    sha256_file,
    short_edge_removed_context,
    split_roles,
    trainable_parameter_count,
    validate_component_disjointness,
    validate_prepared_root,
    validate_selection_receipt,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry, RegistryError  # noqa: E402
from spatial_benchmark.run_archive import RunArchive  # noqa: E402


DEFAULT_PREPARED_ROOT = (
    PROJECT_ROOT
    / "data/processed/same_gene_robustness_v1/variants/v0_within_fov_log1p_all"
)
CONTRACT_PATH = (
    PROJECT_ROOT
    / "experiments/campaigns"
    / CAMPAIGN_ID
    / "frozen_task_contract.yaml"
)
CONTEXT_FILES = {
    "observed_near": "neighbor_near_mean.npy",
    "permuted_near": "neighbor_permuted_near_mean.npy",
    "observed_annular": "neighbor_annular_mean.npy",
}
SLIDES = ("SO_1", "SO_2")
EXPECTED_CELLS = 407_999
EXPECTED_ELIGIBLE = 396_622
EXPECTED_COMPONENTS = 27
SCHEMA_VERSION = 1


class RunnerError(MatchedGraphContextError):
    """Raised when a job violates runner-level lifecycle requirements."""


@dataclass(frozen=True)
class SlideData:
    name: str
    metadata: np.ndarray
    coordinates: np.ndarray
    expression: np.ndarray
    fold: np.ndarray
    component: np.ndarray
    eligible: np.ndarray
    contexts: Mapping[str, np.ndarray]
    near_indptr: np.ndarray
    near_indices: np.ndarray

    def rows(self, folds: Sequence[int]) -> np.ndarray:
        return np.flatnonzero(self.eligible & np.isin(self.fold, tuple(folds)))


@dataclass(frozen=True)
class JobData:
    root: Path
    slides: tuple[SlideData, ...]
    genes: tuple[str, ...]


def _load_array(path: Path, *, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if shape_tail and tuple(value.shape[1:]) != shape_tail:
        raise RunnerError(f"unexpected shape for {path}: {value.shape}")
    return value


def load_job_data(root: Path, *, strict_counts: bool = True) -> JobData:
    validate_prepared_root(root)
    if sha256_file(CONTRACT_PATH) != CONTRACT_SHA256:
        raise RunnerError("frozen contract checksum changed before execution")
    genes_raw = json.loads((root / "genes.json").read_text(encoding="utf-8"))
    if (
        not isinstance(genes_raw, list)
        or len(genes_raw) != GENE_COUNT
        or len(set(map(str, genes_raw))) != GENE_COUNT
    ):
        raise RunnerError("gene authority is not 1,000 ordered unique names")
    slides: list[SlideData] = []
    total_cells = total_eligible = 0
    component_keys: set[tuple[str, int]] = set()
    for slide_name in SLIDES:
        slide_root = root / slide_name
        metadata = _load_array(slide_root / "metadata.npy", shape_tail=(22,))
        coordinates = _load_array(slide_root / "coordinates_um.npy", shape_tail=(2,))
        expression = _load_array(slide_root / "expression_log1p.npy", shape_tail=(GENE_COUNT,))
        fold = _load_array(slide_root / "fold.npy")
        component = _load_array(slide_root / "geometry_group.npy")
        eligible = _load_array(slide_root / "eligible_primary.npy")
        row_count = len(fold)
        arrays = (metadata, coordinates, expression, component, eligible)
        if any(len(value) != row_count for value in arrays):
            raise RunnerError(f"row count mismatch within {slide_name}")
        contexts = {
            arm: _load_array(slide_root / filename, shape_tail=(GENE_COUNT,))
            for arm, filename in CONTEXT_FILES.items()
        }
        if any(len(value) != row_count for value in contexts.values()):
            raise RunnerError(f"context row count mismatch within {slide_name}")
        validate_component_disjointness(fold, component, eligible)
        eligible_components = np.unique(component[np.asarray(eligible, dtype=bool)])
        component_keys.update((slide_name, int(value)) for value in eligible_components)
        total_cells += row_count
        total_eligible += int(np.count_nonzero(eligible))
        slides.append(
            SlideData(
                name=slide_name,
                metadata=metadata,
                coordinates=coordinates,
                expression=expression,
                fold=fold,
                component=component,
                eligible=np.asarray(eligible, dtype=bool),
                contexts=contexts,
                near_indptr=_load_array(slide_root / "near_indptr.npy"),
                near_indices=_load_array(slide_root / "near_indices.npy"),
            )
        )
    if strict_counts and (
        total_cells != EXPECTED_CELLS
        or total_eligible != EXPECTED_ELIGIBLE
        or len(component_keys) != EXPECTED_COMPONENTS
    ):
        raise RunnerError(
            "prepared coverage changed: "
            f"cells={total_cells}, eligible={total_eligible}, components={len(component_keys)}"
        )
    return JobData(root=root, slides=tuple(slides), genes=tuple(map(str, genes_raw)))


def _chunks(rows: np.ndarray, size: int = 8192) -> Iterator[np.ndarray]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _raw_context(
    slide: SlideData,
    rows: np.ndarray,
    *,
    arm: str,
    normalized_base: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    if arm == "no_graph":
        return no_graph_context(normalized_base, projection)
    return np.asarray(slide.contexts[arm][rows], dtype=np.float32)


def fit_preprocessing(
    data: JobData,
    *,
    arm: str,
    training_folds: Sequence[int],
    projection: np.ndarray,
) -> PreprocessingState:
    """Fit every statistic using rows from ``training_folds`` and no others."""

    train_folds = tuple(sorted(map(int, training_folds)))
    coordinate: dict[str, ScaleState] = {}
    for slide in data.slides:
        moments = StreamingMoments(2)
        for rows in _chunks(slide.rows(train_folds)):
            moments.update(np.asarray(slide.coordinates[rows], dtype=np.float32))
        coordinate[slide.name] = moments.finish()

    base_moments = StreamingMoments(BASE_FEATURE_COUNT)
    for slide in data.slides:
        for rows in _chunks(slide.rows(train_folds)):
            base_moments.update(
                raw_base_features(
                    slide.metadata[rows], slide.coordinates[rows], coordinate[slide.name]
                )
            )
    base_state = base_moments.finish()

    target_moments = StreamingMoments(GENE_COUNT)
    context_moments = StreamingMoments(GENE_COUNT)
    for slide in data.slides:
        for rows in _chunks(slide.rows(train_folds)):
            raw_base = raw_base_features(
                slide.metadata[rows], slide.coordinates[rows], coordinate[slide.name]
            )
            base = base_state.transform(raw_base)
            target_moments.update(np.asarray(slide.expression[rows], dtype=np.float32))
            context_moments.update(
                _raw_context(slide, rows, arm=arm, normalized_base=base, projection=projection)
            )
    return PreprocessingState(
        coordinate=coordinate,
        base=base_state,
        target=target_moments.finish(),
        context=context_moments.finish(),
        projection_sha256=ndarray_sha256(projection),
        training_folds=train_folds,
    )


def _batch(
    slide: SlideData,
    rows: np.ndarray,
    *,
    arm: str,
    preprocessing: PreprocessingState,
    projection: np.ndarray,
    permitted_target_folds: Sequence[int],
    context_variant: str = "native",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    actual_folds = set(map(int, np.unique(slide.fold[rows])))
    if not actual_folds.issubset(set(map(int, permitted_target_folds))):
        raise RunnerError("attempted to read outcomes outside this job's permitted folds")
    base = preprocessing.base.transform(
        raw_base_features(
            slide.metadata[rows],
            slide.coordinates[rows],
            preprocessing.coordinate[slide.name],
        )
    )
    degree: np.ndarray | None = None
    if context_variant == "native":
        raw_context = _raw_context(
            slide, rows, arm=arm, normalized_base=base, projection=projection
        )
        context = preprocessing.context.transform(raw_context)
    elif context_variant == "zero":
        context = np.zeros((len(rows), GENE_COUNT), dtype=np.float32)
    elif context_variant in {"permuted_near", "observed_annular"}:
        context = preprocessing.context.transform(
            np.asarray(slide.contexts[context_variant][rows], dtype=np.float32)
        )
    elif context_variant == "observed_near_10_25":
        raw_context, degree = short_edge_removed_context(
            receiver_indices=rows,
            coordinates=slide.coordinates,
            indptr=slide.near_indptr,
            indices=slide.near_indices,
            source_expression=slide.expression,
        )
        context = preprocessing.context.transform(raw_context)
        # Contract semantics are literal at the learned input: if filtering
        # removes every frozen source, the receiver contributes no context.
        context[np.asarray(degree) == 0] = 0.0
    else:
        raise RunnerError(f"unknown context substitution {context_variant!r}")
    target = preprocessing.target.transform(
        np.asarray(slide.expression[rows], dtype=np.float32)
    )
    return base, context, target, degree


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _device(value: str) -> torch.device:
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RunnerError("CUDA was requested but is not available")
    return selected


def train_epoch(
    model: MatchedAdditiveContextMLP,
    optimizer: torch.optim.Optimizer,
    data: JobData,
    *,
    arm: str,
    preprocessing: PreprocessingState,
    projection: np.ndarray,
    training_folds: Sequence[int],
    batch_size: int,
    seed: int,
    epoch: int,
    device: torch.device,
) -> float:
    model.train()
    generator = np.random.default_rng(seed + epoch * 1_000_003)
    slide_order = generator.permutation(len(data.slides))
    squared_error = 0.0
    element_count = 0
    for slide_index in slide_order:
        slide = data.slides[int(slide_index)]
        rows = slide.rows(training_folds)
        rows = rows[generator.permutation(len(rows))]
        for selected in _chunks(rows, batch_size):
            base, context, target, _ = _batch(
                slide,
                selected,
                arm=arm,
                preprocessing=preprocessing,
                projection=projection,
                permitted_target_folds=training_folds,
            )
            base_tensor = torch.from_numpy(base).to(device)
            context_tensor = torch.from_numpy(context).to(device)
            target_tensor = torch.from_numpy(target).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(base_tensor, context_tensor)
            loss = torch.mean(torch.square(prediction - target_tensor))
            if not torch.isfinite(loss):
                raise RunnerError("training produced a nonfinite loss")
            loss.backward()
            optimizer.step()
            squared_error += float(loss.detach().cpu()) * target_tensor.numel()
            element_count += target_tensor.numel()
    return squared_error / element_count


def _gelu_derivative(value: torch.Tensor) -> torch.Tensor:
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    inverse_sqrt_two_pi = 1.0 / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + torch.erf(value * inverse_sqrt_two)) + value * torch.exp(
        -0.5 * value * value
    ) * inverse_sqrt_two_pi


@torch.no_grad()
def evaluate(
    model: MatchedAdditiveContextMLP,
    data: JobData,
    *,
    arm: str,
    preprocessing: PreprocessingState,
    projection: np.ndarray,
    evaluation_fold: int,
    batch_size: int,
    device: torch.device,
    context_variant: str = "native",
    collect_hidden_derivative: bool = False,
) -> tuple[MetricAccumulator, dict[tuple[str, int], tuple[np.ndarray, int]], int]:
    model.eval()
    accumulator = MetricAccumulator(GENE_COUNT)
    derivatives: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
    zero_degree_count = 0
    for slide in data.slides:
        for rows in _chunks(slide.rows((evaluation_fold,)), batch_size):
            base, context, target, degree = _batch(
                slide,
                rows,
                arm=arm,
                preprocessing=preprocessing,
                projection=projection,
                permitted_target_folds=(evaluation_fold,),
                context_variant=context_variant,
            )
            base_tensor = torch.from_numpy(base).to(device)
            context_tensor = torch.from_numpy(context).to(device)
            prediction = model(base_tensor, context_tensor)
            prediction_numpy = prediction.cpu().numpy()
            components = np.asarray(slide.component[rows])
            accumulator.update(
                target, prediction_numpy, slide=slide.name, components=components
            )
            if degree is not None:
                zero_degree_count += int(np.count_nonzero(degree == 0))
            if collect_hidden_derivative:
                derivative = _gelu_derivative(model.context_hidden(context_tensor)).cpu().numpy()
                for component in np.unique(components):
                    mask = components == component
                    key = (slide.name, int(component))
                    prior_sum, prior_n = derivatives.get(
                        key,
                        (np.zeros(model.hidden_width, dtype=np.float64), 0),
                    )
                    derivatives[key] = (
                        prior_sum + derivative[mask].sum(axis=0, dtype=np.float64),
                        prior_n + int(mask.sum()),
                    )
    return accumulator, derivatives, zero_degree_count


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    materialized = list(rows)
    if not materialized:
        raise RunnerError("refusing to write an empty result table")
    return ("\n".join(canonical_json(row) for row in materialized) + "\n").encode("utf-8")


def _output_entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_kind(relative: Path) -> str:
    if relative.as_posix() == "results.json":
        return "campaign_result"
    if relative.parts and relative.parts[0] == "checkpoints":
        return "checkpoint"
    if relative.parts and relative.parts[0] == "metrics":
        return "metrics"
    if relative.parts and relative.parts[0] == "provenance":
        return "provenance"
    if relative.parts and relative.parts[0] == "interpretation":
        return "interpretation"
    return "run_file"


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.is_symlink() or path.name.startswith("_"):
            continue
        records.append(
            {
                "kind": _artifact_kind(path.relative_to(root)),
                "path": path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _normalization_npz(preprocessing: PreprocessingState) -> bytes:
    stream = io.BytesIO()
    values: dict[str, np.ndarray] = {
        "base_mean": preprocessing.base.mean,
        "base_scale": preprocessing.base.scale,
        "target_mean": preprocessing.target.mean,
        "target_scale": preprocessing.target.scale,
        "context_mean": preprocessing.context.mean,
        "context_scale": preprocessing.context.scale,
    }
    for slide, state in preprocessing.coordinate.items():
        values[f"coordinate_mean_{slide}"] = state.mean
        values[f"coordinate_scale_{slide}"] = state.scale
    np.savez(stream, **values)
    return stream.getvalue()


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json(list(array.shape)).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _checkpoint_bytes(
    model: MatchedAdditiveContextMLP,
    *,
    config: Mapping[str, Any],
    preprocessing: PreprocessingState,
    selection_receipt: Mapping[str, Any],
) -> tuple[bytes, str, dict[str, Any]]:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state_hash = _state_dict_sha256(state)
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "contract_sha256": CONTRACT_SHA256,
        "model_config": dict(config),
        "preprocessing": preprocessing.payload(),
        "selection_payload_sha256": selection_receipt["payload_sha256"],
        "state_dict_sha256": state_hash,
        "state_dict": state,
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    # Immediate CPU replay catches corrupt serialization and architecture drift.
    replay = torch.load(io.BytesIO(stream.getvalue()), map_location="cpu", weights_only=False)
    replay_model = build_matched_model(
        "no_graph",
        hidden_width=int(config["hidden_width"]),
        dropout=float(config["dropout"]),
        seed=0,
    )
    replay_model.load_state_dict(replay["state_dict"], strict=True)
    if _state_dict_sha256(replay_model.state_dict()) != state_hash:
        raise RunnerError("checkpoint replay checksum failed")
    original_model = build_matched_model(
        "no_graph",
        hidden_width=int(config["hidden_width"]),
        dropout=float(config["dropout"]),
        seed=0,
    )
    original_model.load_state_dict(state, strict=True)
    original_model.eval()
    replay_model.eval()
    generator = torch.Generator(device="cpu").manual_seed(2026081202)
    replay_base = torch.randn((3, BASE_FEATURE_COUNT), generator=generator)
    replay_context = torch.randn((3, GENE_COUNT), generator=generator)
    with torch.no_grad():
        expected = original_model(replay_base, replay_context)
        actual = replay_model(replay_base, replay_context)
    if (
        not torch.isfinite(expected).all()
        or not torch.isfinite(actual).all()
        or not torch.equal(expected, actual)
    ):
        raise RunnerError("checkpoint forward replay failed")
    replay_output = np.ascontiguousarray(actual.numpy())
    replay_diagnostic = {
        "status": "passed",
        "finite": True,
        "bitwise_equal": True,
        "sample_shape": list(replay_output.shape),
        "output_sha256": ndarray_sha256(replay_output),
    }
    return stream.getvalue(), state_hash, replay_diagnostic


def _context_jacobian_bytes(
    model: MatchedAdditiveContextMLP,
    derivatives: Mapping[tuple[str, int], tuple[np.ndarray, int]],
    genes: Sequence[str],
) -> tuple[bytes, list[dict[str, Any]]]:
    if not derivatives:
        raise RunnerError("context Jacobian requested without evaluation derivatives")
    component_rows: list[dict[str, Any]] = []
    component_means = []
    for (slide, component), (derivative_sum, count) in sorted(derivatives.items()):
        mean_derivative = derivative_sum / count
        component_means.append(mean_derivative)
        component_rows.append(
            {
                "slide": slide,
                "component": component,
                "n_cells": count,
                "mean_hidden_derivative": mean_derivative.tolist(),
            }
        )
    equal_component_derivative = np.mean(component_means, axis=0)
    linear = model.context_linear.weight.detach().cpu().numpy().astype(np.float32)
    hidden = model.context_hidden.weight.detach().cpu().numpy().astype(np.float32)
    output = model.context_output.weight.detach().cpu().numpy().astype(np.float32)
    nonlinear = (output * equal_component_derivative[None, :]) @ hidden
    nonlinear = np.asarray(nonlinear, dtype=np.float32)
    total = np.asarray(linear + nonlinear, dtype=np.float32)
    stream = io.BytesIO()
    np.savez(
        stream,
        total=total,
        linear=linear,
        nonlinear=nonlinear,
        mean_hidden_derivative=np.asarray(equal_component_derivative, dtype=np.float32),
        genes=np.asarray(genes, dtype="U"),
        eligible_gene_mask=np.ones(len(genes), dtype=bool),
    )
    return stream.getvalue(), component_rows


def _git_provenance() -> dict[str, Any]:
    def command(*arguments: str) -> str:
        return subprocess.run(
            arguments,
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()

    status = command("git", "status", "--porcelain")
    return {
        "commit": command("git", "rev-parse", "HEAD") or "unknown",
        "dirty": bool(status),
        "dirty_fingerprint": hashlib.sha256(status.encode("utf-8")).hexdigest() if status else None,
    }


def _hardware() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name() if cuda else None,
    }


def _register_start(
    registry: Registry,
    *,
    run_id: str,
    config: Mapping[str, Any],
    seed: int,
    fold: int,
    attempt: int,
) -> None:
    if registry.get_campaign(CAMPAIGN_ID) is None:
        registry.create_campaign(
            CAMPAIGN_ID,
            name="Matched graph-context nested-CV experiment",
            scientific_question="Does aligned local RNA context improve whole-node prediction?",
            config={"contract_sha256": CONTRACT_SHA256},
        )
    variant_id = scientific_id(config)
    registry.register_variant(
        variant_id,
        campaign_id=CAMPAIGN_ID,
        configuration=config,
        model_family="matched_additive_context_mlp",
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        split_id=SPLIT_ID,
        embedding_dim=int(config["model"]["hidden_width"]),
        learning_rate=float(config["trainer"]["learning_rate"]),
        batch_size=int(config["trainer"]["batch_size"]),
    )
    row = registry.get_run(run_id)
    if row is None:
        registry.create_run(
            run_id,
            campaign_id=CAMPAIGN_ID,
            scientific_id=variant_id,
            repro_id=f"rep_{canonical_sha256({'config': config, 'contract': CONTRACT_SHA256})[:20]}",
            seed=seed,
            fold=fold,
            attempt=attempt,
            configuration=config,
            status="pending",
            model_family="matched_additive_context_mlp",
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            split_id=SPLIT_ID,
            preprocessing_version=PROCESSED_FINGERPRINT,
            dataset_fingerprint=PROCESSED_FINGERPRINT,
            split_fingerprint=SPLIT_FINGERPRINT,
        )
    current = registry.get_run(run_id)
    assert current is not None
    if current["status"] == "pending":
        registry.transition_run(run_id, "running", start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    elif current["status"] != "running":
        raise RunnerError(f"registry run is not runnable: {current['status']}")


def _archive_for(run_id: str, scratch_run: Path | None) -> RunArchive:
    paths = current_paths()
    expected = paths.scratch_root / "active_runs" / run_id
    if expected.exists():
        return RunArchive.attach_active(run_id, paths=paths, scratch_path=scratch_run)
    if scratch_run is not None and scratch_run.resolve(strict=False) != expected.resolve(strict=False):
        raise RunnerError("--scratch-run must be the canonical active run path")
    return RunArchive.create(run_id, paths=paths)


def _write_jsonl(archive: RunArchive, relative: str, rows: Iterable[Mapping[str, Any]]) -> Path:
    return archive.write_bytes(relative, _jsonl_bytes(rows))


def _base_config(
    *,
    mode: str,
    arm: str,
    fold: int,
    seed: int,
    batch_size: int,
    model_config: Mapping[str, Any],
    roles: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = (
        {"role": "validation", "primary_metric": "validation/component_equal_mse"}
        if mode == "tune"
        else {
            "role": "test",
            "statistical_partition": "outer_geometry_test",
            "primary_metric": "test/component_equal_mse",
        }
    )
    return {
        "campaign_id": CAMPAIGN_ID,
        "contract_sha256": CONTRACT_SHA256,
        "mode": mode,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "dataset": {
            "dataset_id": DATASET_ID,
            "version": DATASET_VERSION,
            "split_id": SPLIT_ID,
            "processed_fingerprint": PROCESSED_FINGERPRINT,
            "split_fingerprint": SPLIT_FINGERPRINT,
        },
        "split_roles": dict(roles),
        "model": {
            "family": "matched_additive_context_mlp",
            "hidden_width": int(model_config["hidden_width"]),
            "dropout": float(model_config["dropout"]),
            "gene_count": GENE_COUNT,
            "base_feature_count": BASE_FEATURE_COUNT,
        },
        "trainer": {
            "optimizer": "AdamW",
            "learning_rate": float(model_config["learning_rate"]),
            "weight_decay": float(model_config["weight_decay"]),
            "batch_size": batch_size,
            "epoch": int(model_config.get("epoch", max(EVALUATION_EPOCHS))),
            "precision": "fp32",
            "deterministic": True,
        },
        "evaluation": evaluation,
    }


def run_real_job(args: argparse.Namespace) -> Path:
    start = time.monotonic()
    if args.arm not in ARMS or args.fold not in range(4):
        raise RunnerError("real jobs require a frozen arm and fold 0 through 3")
    roles = split_roles(args.mode, args.fold)
    if args.mode == "tune":
        if args.seed not in TUNING_SEEDS:
            raise RunnerError("tuning seed is outside the frozen Stage A/B set")
        if not args.candidate_id or args.selection_receipt is not None:
            raise RunnerError("tune requires --candidate-id and prohibits a selection receipt")
        model_config = candidate_config(args.candidate_id)
        max_epochs = max(EVALUATION_EPOCHS)
        evaluation_epochs = EVALUATION_EPOCHS
        selection_payload = None
        selection_binding = None
    else:
        if args.seed not in CONFIRMATION_SEEDS:
            raise RunnerError("confirmation seed is outside the frozen five-seed set")
        if args.candidate_id is not None or args.selection_receipt is None:
            raise RunnerError("confirm requires --selection-receipt and prohibits candidate selection")
        selection_payload, model_config = validate_selection_receipt(
            args.selection_receipt, arm=args.arm, outer_fold=args.fold
        )
        max_epochs = int(model_config["epoch"])
        evaluation_epochs = (max_epochs,)
        selection_binding = {
            "path": args.selection_receipt.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(args.selection_receipt),
            "payload_sha256": selection_payload["payload_sha256"],
        }
    if args.allow_test_overrides:
        max_epochs = int(args.max_epochs or max_epochs)
        evaluation_epochs = tuple(args.evaluation_epochs or (max_epochs,))
    elif args.max_epochs is not None or args.evaluation_epochs is not None or args.batch_size != 4096:
        raise RunnerError("production hyperparameters are frozen; test overrides require explicit opt-in")
    if any(epoch < 1 or epoch > max_epochs for epoch in evaluation_epochs):
        raise RunnerError("evaluation epoch falls outside the continuous trajectory")

    if not args.allow_test_overrides:
        free_gib = shutil.disk_usage(current_paths().project_root).free / 1024**3
        if free_gib < 25.0:
            raise RunnerError(f"free disk {free_gib:.2f} GiB is below the 25 GiB stop gate")

    archive = _archive_for(args.run_id, args.scratch_run)
    config = _base_config(
        mode=args.mode,
        arm=args.arm,
        fold=args.fold,
        seed=args.seed,
        batch_size=args.batch_size,
        model_config={**model_config, "epoch": max_epochs},
        roles=roles,
    )
    registry = None if args.no_registry else Registry(args.registry)
    if registry is not None:
        _register_start(
            registry,
            run_id=args.run_id,
            config=config,
            seed=args.seed,
            fold=args.fold,
            attempt=args.attempt,
        )

    prepared_root = args.prepared_root.resolve()
    data = load_job_data(prepared_root, strict_counts=not args.allow_test_overrides)
    projection = frozen_rank27_projection()
    preprocessing = fit_preprocessing(
        data,
        arm=args.arm,
        training_folds=roles["train_folds"],
        projection=projection,
    )
    device = _device(args.device)
    _set_determinism(args.seed)
    model = build_matched_model(
        args.arm,
        hidden_width=int(model_config["hidden_width"]),
        dropout=float(model_config["dropout"]),
        seed=args.seed,
    ).to(device)
    parameter_count = trainable_parameter_count(model)
    # Literal four-arm construction is a cheap runtime guard against future arm branches.
    counts = {
        trainable_parameter_count(
            build_matched_model(
                arm,
                hidden_width=int(model_config["hidden_width"]),
                dropout=float(model_config["dropout"]),
                seed=args.seed,
            )
        )
        for arm in ARMS
    }
    if counts != {parameter_count}:
        raise RunnerError("trainable parameter count differs among arms")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    validation_by_epoch: list[dict[str, Any]] = []
    tuning_component_rows: list[dict[str, Any]] = []
    final_accumulator: MetricAccumulator | None = None
    evaluation_fold = int(
        roles["validation_fold"] if args.mode == "tune" else roles["test_fold"]
    )
    for epoch in range(1, max_epochs + 1):
        train_mse = train_epoch(
            model,
            optimizer,
            data,
            arm=args.arm,
            preprocessing=preprocessing,
            projection=projection,
            training_folds=roles["train_folds"],
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
            device=device,
        )
        history.append({"epoch": epoch, "train_mse": train_mse})
        if epoch in evaluation_epochs:
            accumulator, _, _ = evaluate(
                model,
                data,
                arm=args.arm,
                preprocessing=preprocessing,
                projection=projection,
                evaluation_fold=evaluation_fold,
                batch_size=args.batch_size,
                device=device,
            )
            mse, mae = accumulator.component_equal()
            final_accumulator = accumulator
            if args.mode == "tune":
                validation_by_epoch.append(
                    {
                        "epoch": epoch,
                        "validation_component_equal_mse": mse,
                        "validation_component_equal_mae": mae,
                    }
                )
                tuning_component_rows.extend(
                    accumulator.component_rows(
                        run_id=args.run_id,
                        arm=args.arm,
                        fold=args.fold,
                        seed=args.seed,
                        epoch=epoch,
                        split="validation",
                    )
                )
    if final_accumulator is None:
        raise RunnerError("continuous training trajectory produced no evaluation")
    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    if not args.allow_test_overrides and peak_vram_gb > 20.5:
        raise RunnerError(
            f"peak VRAM {peak_vram_gb:.3f} GiB exceeds the 20.5 GiB stop gate"
        )

    archive.write_resolved_config(config)
    archive.write_bytes("provenance/normalization.npz", _normalization_npz(preprocessing))
    archive.write_json("provenance/normalization.json", preprocessing.payload())
    archive.write_json(
        "provenance/data_fingerprints.json",
        {
            "prepared_manifest_sha256": PREPARED_MANIFEST_SHA256,
            "integrity_manifest_sha256": INTEGRITY_MANIFEST_SHA256,
            "processed_fingerprint": PROCESSED_FINGERPRINT,
            "split_fingerprint": SPLIT_FINGERPRINT,
        },
    )
    archive.write_json("provenance/git.json", _git_provenance())
    archive.write_json("provenance/hardware.json", _hardware())
    archive.write_text("provenance/environment.txt", "\n".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith(("CUDA", "CUBLAS", "PYTHON"))) + "\n")
    archive.write_text("provenance/command.txt", " ".join(sys.argv) + "\n")
    archive.write_text("logs/stdout.log", "matched graph-context job completed\n")
    archive.write_text("logs/stderr.log", "")
    history_path = _write_jsonl(archive, "metrics/history.jsonl", history)
    outputs: dict[str, Any] = {"history": _output_entry(history_path, archive.scratch_path)}

    if args.mode == "tune":
        by_epoch_path = _write_jsonl(
            archive, "metrics/validation_by_epoch.jsonl", validation_by_epoch
        )
        component_path = _write_jsonl(
            archive, "metrics/component_metrics.jsonl", tuning_component_rows
        )
        last = validation_by_epoch[-1]
        metrics = {
            "validation_component_equal_mse": last["validation_component_equal_mse"],
            "validation_component_equal_mae": last["validation_component_equal_mae"],
        }
        final_metrics = {
            "validation/component_equal_mse": metrics["validation_component_equal_mse"],
            "validation/component_equal_mae": metrics["validation_component_equal_mae"],
        }
        outputs.update(
            validation_by_epoch=_output_entry(by_epoch_path, archive.scratch_path),
            component_metrics=_output_entry(component_path, archive.scratch_path),
        )
        coverage_complete = len(validation_by_epoch) == len(evaluation_epochs)
    else:
        native = final_accumulator
        primary_mse, primary_mae = native.component_equal()
        component_rows = native.component_rows(
            run_id=args.run_id,
            arm=args.arm,
            fold=args.fold,
            seed=args.seed,
            split="test",
            context_variant="native",
        )
        gene_rows = native.gene_rows(
            data.genes,
            run_id=args.run_id,
            arm=args.arm,
            fold=args.fold,
            seed=args.seed,
            split="test",
            context_variant="native",
        )
        component_gene_rows = native.component_gene_rows(
            data.genes,
            run_id=args.run_id,
            arm=args.arm,
            fold=args.fold,
            seed=args.seed,
        )
        component_path = _write_jsonl(
            archive, "metrics/component_metrics.jsonl", component_rows
        )
        gene_path = _write_jsonl(archive, "metrics/gene_metrics.jsonl", gene_rows)
        component_gene_path = _write_jsonl(
            archive,
            "metrics/component_gene_metrics.jsonl",
            component_gene_rows,
        )
        checkpoint, state_hash, checkpoint_replay = _checkpoint_bytes(
            model,
            config=model_config,
            preprocessing=preprocessing,
            selection_receipt=selection_payload,
        )
        checkpoint_path = archive.write_bytes("checkpoints/last.ckpt", checkpoint)
        archive.copy_file(args.selection_receipt, "provenance/selection_receipt.json")
        outputs.update(
            component_metrics=_output_entry(component_path, archive.scratch_path),
            gene_metrics=_output_entry(gene_path, archive.scratch_path),
            component_gene_metrics=_output_entry(
                component_gene_path, archive.scratch_path
            ),
            checkpoint=_output_entry(checkpoint_path, archive.scratch_path),
        )
        outputs["checkpoint"]["state_dict_sha256"] = state_hash
        metrics = {
            "test_component_equal_mse": primary_mse,
            "test_component_equal_mae": primary_mae,
        }
        final_metrics = {
            "test/component_equal_mse": primary_mse,
            "test/component_equal_mae": primary_mae,
        }
        expected_components = {(row["slide"], row["component"]) for row in component_rows}
        component_gene_coverage = {
            (row["slide"], row["component"], row["gene_index"])
            for row in component_gene_rows
        }
        expected_component_gene_coverage = {
            (slide, component, gene_index)
            for slide, component in expected_components
            for gene_index in range(GENE_COUNT)
        }
        coverage_complete = (
            len(gene_rows) == GENE_COUNT
            and bool(expected_components)
            and component_gene_coverage == expected_component_gene_coverage
        )
        if args.arm == "observed_near":
            substitution_component_rows: list[dict[str, Any]] = []
            substitution_gene_rows: list[dict[str, Any]] = []
            substitution_component_gene_rows: list[dict[str, Any]] = []
            zero_degree: dict[str, int] = {}
            native_derivatives: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
            for variant in FAITHFULNESS_CONTEXTS:
                if variant == "native":
                    accumulator, derivatives, zero_count = evaluate(
                        model,
                        data,
                        arm=args.arm,
                        preprocessing=preprocessing,
                        projection=projection,
                        evaluation_fold=evaluation_fold,
                        batch_size=args.batch_size,
                        device=device,
                        context_variant=variant,
                        collect_hidden_derivative=True,
                    )
                    native_derivatives = derivatives
                else:
                    accumulator, _, zero_count = evaluate(
                        model,
                        data,
                        arm=args.arm,
                        preprocessing=preprocessing,
                        projection=projection,
                        evaluation_fold=evaluation_fold,
                        batch_size=args.batch_size,
                        device=device,
                        context_variant=variant,
                    )
                zero_degree[variant] = zero_count
                substitution_component_rows.extend(
                    accumulator.component_rows(
                        run_id=args.run_id,
                        arm=args.arm,
                        fold=args.fold,
                        seed=args.seed,
                        split="test",
                        context_variant=variant,
                    )
                )
                substitution_gene_rows.extend(
                    accumulator.gene_rows(
                        data.genes,
                        run_id=args.run_id,
                        arm=args.arm,
                        fold=args.fold,
                        seed=args.seed,
                        split="test",
                        context_variant=variant,
                    )
                )
                substitution_component_gene_rows.extend(
                    accumulator.component_gene_rows(
                        data.genes,
                        run_id=args.run_id,
                        arm=args.arm,
                        fold=args.fold,
                        seed=args.seed,
                        context_variant=variant,
                    )
                )
            substitution_path = _write_jsonl(
                archive,
                "metrics/context_substitution.jsonl",
                substitution_component_rows,
            )
            substitution_gene_path = _write_jsonl(
                archive,
                "metrics/context_substitution_gene_metrics.jsonl",
                substitution_gene_rows,
            )
            substitution_component_gene_path = _write_jsonl(
                archive,
                "metrics/context_substitution_component_gene_metrics.jsonl",
                substitution_component_gene_rows,
            )
            jacobian_bytes, derivative_rows = _context_jacobian_bytes(
                model, native_derivatives, data.genes
            )
            jacobian_path = archive.write_bytes(
                "interpretation/context_jacobian.npz", jacobian_bytes
            )
            derivative_path = _write_jsonl(
                archive,
                "interpretation/component_hidden_derivatives.jsonl",
                derivative_rows,
            )
            archive.write_json(
                "diagnostics/short_edge_sensitivity.json",
                {
                    "rule": "filter_frozen_near_csr_endpoints_by_euclidean_distance_no_new_edges_zero_context_if_empty",
                    "minimum_um": 10.0,
                    "maximum_um": 25.0,
                    "zero_degree_receivers": zero_degree["observed_near_10_25"],
                    "zero_degree_context": "all_zero_vector_after_train_fitted_observed_near_scaling",
                },
            )
            outputs.update(
                context_substitution=_output_entry(substitution_path, archive.scratch_path),
                context_substitution_gene_metrics=_output_entry(
                    substitution_gene_path, archive.scratch_path
                ),
                context_substitution_component_gene_metrics=_output_entry(
                    substitution_component_gene_path, archive.scratch_path
                ),
                context_jacobian=_output_entry(jacobian_path, archive.scratch_path),
                component_hidden_derivatives=_output_entry(
                    derivative_path, archive.scratch_path
                ),
            )
            actual_coverage = {
                (row["slide"], row["component"], row["context_variant"])
                for row in substitution_component_rows
            }
            expected_coverage = {
                (slide, component, variant)
                for slide, component in expected_components
                for variant in FAITHFULNESS_CONTEXTS
            }
            coverage_complete = coverage_complete and actual_coverage == expected_coverage

    archive.write_json("metrics/final.json", final_metrics)
    for name, value in final_metrics.items():
        archive.append_metric_event({"name": name, "value": value, "step": max_epochs})
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "finite_metrics": all(math.isfinite(float(value)) for value in final_metrics.values()),
        "coverage_complete": bool(coverage_complete),
        "campaign_id": CAMPAIGN_ID,
        "contract_sha256": CONTRACT_SHA256,
        "mode": args.mode,
        "run_id": args.run_id,
        "arm": args.arm,
        "fold": args.fold,
        "seed": args.seed,
        "candidate_id": model_config.get("candidate_id"),
        "config": dict(model_config),
        "config_sha256": canonical_sha256(model_config),
        "resolved_config_sha256": canonical_sha256(config),
        "input": {
            "manifest_sha256": PREPARED_MANIFEST_SHA256,
            "prepared_manifest_sha256": PREPARED_MANIFEST_SHA256,
            "integrity_manifest_sha256": INTEGRITY_MANIFEST_SHA256,
            "processed_fingerprint": PROCESSED_FINGERPRINT,
            "split_fingerprint": SPLIT_FINGERPRINT,
        },
        "split_roles": roles,
        "parameter_count": parameter_count,
        "normalization_sha256": preprocessing.payload()["state_sha256"],
        "projection_sha256": ndarray_sha256(projection),
        "metrics": metrics,
        "outputs": outputs,
        "runtime_seconds": time.monotonic() - start,
        "peak_vram_gb": peak_vram_gb,
        "peak_host_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    if args.mode == "tune":
        results["validation_by_epoch"] = validation_by_epoch
        results["selected_epoch"] = None
    else:
        results["selected_epoch"] = max_epochs
        results["selection_receipt"] = selection_binding
        results["checkpoint_replay"] = checkpoint_replay
    if not results["finite_metrics"] or not results["coverage_complete"]:
        raise RunnerError("result failed finite-metric or coverage checks")
    archive.write_json("results.json", results)
    archive.write_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            "status": "success",
            "campaign_id": CAMPAIGN_ID,
            "mode": args.mode,
            "arm": args.arm,
            "contract_sha256": CONTRACT_SHA256,
        }
    )
    archive.write_summary(
        {
            "status": "success",
            "mode": args.mode,
            "arm": args.arm,
            "primary_metric_name": next(iter(final_metrics)),
            "primary_metric_value": next(iter(final_metrics.values())),
            "parameter_count": parameter_count,
        }
    )
    if not args.allow_test_overrides:
        remaining_gib = shutil.disk_usage(current_paths().project_root).free / 1024**3
        if remaining_gib < 25.0:
            raise RunnerError(
                f"free disk {remaining_gib:.2f} GiB fell below the 25 GiB publish gate"
            )
    # Campaign-specific lifecycle: the generic prediction/checkpoint contract is
    # intentionally inapplicable to validation-only tuning bundles.
    archive._write_checksums()
    published = archive._publish_unmarked()
    if registry is not None:
        try:
            registry.transition_run(
                args.run_id,
                "finalizing",
                duration_seconds=time.monotonic() - start,
                host=socket.gethostname(),
                gpu_model=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                peak_vram_gb=results["peak_vram_gb"],
                parameter_count=parameter_count,
                primary_metric_name=next(iter(final_metrics)),
                primary_metric_value=next(iter(final_metrics.values())),
                artifact_path=published,
            )
            for name, value in final_metrics.items():
                registry.record_metric(
                    args.run_id,
                    name,
                    value,
                    step=max_epochs,
                    split="validation" if args.mode == "tune" else "test",
                )
            registry.record_artifacts(args.run_id, _artifact_records(published))
            registry.transition_run(
                args.run_id,
                "completed",
                end_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        except BaseException:
            current = registry.get_run(args.run_id)
            if current and current["status"] in {"running", "finalizing"}:
                registry.transition_run(
                    args.run_id,
                    "failed",
                    end_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    artifact_path=published,
                    failure_category="artifact_registry_failure",
                )
            archive._mark_published("_FAILED")
            raise
    archive._mark_published("_SUCCESS")
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("tune", "confirm", "synthetic-gate"))
    parser.add_argument("--run-id")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--candidate-id")
    parser.add_argument("--selection-receipt", type=Path)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--scratch-run", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--no-registry", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--evaluation-epochs", type=lambda value: tuple(map(int, value.split(","))))
    parser.add_argument("--allow-test-overrides", action="store_true", help=argparse.SUPPRESS)
    return parser


def _preserve_failed_job(args: argparse.Namespace, error: BaseException) -> None:
    """Best-effort failure preservation without masking the scientific error."""

    if not args.run_id:
        return
    paths = current_paths()
    scratch = paths.scratch_root / "active_runs" / args.run_id
    published = RunArchive.artifact_path_for(args.run_id, paths)
    final_path: Path | None = None
    registry: Registry | None = None
    registry_row: Mapping[str, Any] | None = None
    if not args.no_registry:
        try:
            registry = Registry(args.registry)
            registry_row = registry.get_run(args.run_id)
        except BaseException:
            registry = None
            registry_row = None
    try:
        if scratch.is_dir():
            archive = RunArchive.attach_active(
                args.run_id, paths=paths, scratch_path=scratch
            )
            # A failed publish may leave a now-stale generated checksum authority;
            # failure finalization must recompute it after adding diagnostics.
            checksum = scratch / "provenance/artifact_checksums.json"
            if checksum.is_file() and not checksum.is_symlink():
                checksum.unlink()
            final_path = archive.finalize_failure(
                error, failure_category="matched_graph_context_failure"
            )
        elif published.is_dir():
            markers = [
                marker
                for marker in ("_SUCCESS", "_FAILED", "_PRUNED")
                if (published / marker).exists()
            ]
            if not markers:
                archive = RunArchive.from_published(args.run_id, paths=paths)
                if registry_row and registry_row.get("status") == "completed":
                    final_path = archive._mark_published("_SUCCESS")
                else:
                    final_path = archive._mark_published("_FAILED")
            else:
                final_path = published
    except BaseException:
        final_path = published if published.is_dir() else None

    if registry is None:
        return
    try:
        row = registry.get_run(args.run_id)
        if row and row["status"] in {"pending", "running", "finalizing"}:
            registry.transition_run(
                args.run_id,
                "failed",
                end_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                artifact_path=final_path,
                failure_category="matched_graph_context_failure",
            )
            if final_path is not None:
                registry.record_artifacts(args.run_id, _artifact_records(final_path))
    except BaseException:
        # The original exception remains authoritative; coordinator reconciliation
        # can recover an immutable failed bundle or a still-running registry row.
        return


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "synthetic-gate":
        result = run_synthetic_recovery_gates(args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 2
    if not args.run_id or args.arm is None or args.fold is None:
        raise RunnerError("real-data modes require --run-id, --arm, and --fold")
    try:
        artifact = run_real_job(args)
    except BaseException as error:
        _preserve_failed_job(args, error)
        raise
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

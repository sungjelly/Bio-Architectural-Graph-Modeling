"""Selection, aggregation, and durability helpers for the SO2 NB2 run.

The production training entry point lives in
``scripts/train/run_so2_geometry_modulated_nb.py``.  This module keeps the
parts that must be independently testable free of CUDA and distributed side
effects: validation aggregation, early stopping, epoch-metric durability, and
the bounded latest/best checkpoint lifecycle.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROTOCOL = "donor_grouped_so2_geometry_modulated_nb_train12_val2_earlystop_v1"
CHECKPOINT_SCHEMA = "so2_geometry_modulated_nb_train12_val2_checkpoint_v1"
EPOCH_METRICS_SCHEMA = "so2_geometry_modulated_nb_epoch_metrics_v1"
TRAINING_ALIASES = tuple(f"SO2-C{number:02d}" for number in range(15, 27))
VALIDATION_ALIASES = ("SO2-C27", "SO2-C28")
VALIDATION_VIEWS_PER_CORE = 10
OPTIMIZER_UPDATES_PER_EPOCH = 6
EXPECTED_PARAMETER_COUNT = 5_135_088


class SO2NBTrainingError(RuntimeError):
    """Raised when the frozen NB2 training contract would be ambiguous."""


def _finite(value: Any, *, field: str, nonnegative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = " finite and non-negative" if nonnegative else " finite"
        raise SO2NBTrainingError(f"{field} must be{qualifier}.")
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EarlyStoppingState:
    """Complete rank-independent state for validation-controlled stopping."""

    best_value: float | None = None
    best_epoch: int = 0
    bad_validations: int = 0
    completed_epoch: int = 0
    improved: bool = False
    should_stop: bool = False
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.best_value is not None:
            _finite(self.best_value, field="best_value")
        for name in ("best_epoch", "bad_validations", "completed_epoch"):
            if int(getattr(self, name)) < 0:
                raise SO2NBTrainingError(f"{name} cannot be negative.")
        if int(self.best_epoch) > int(self.completed_epoch):
            raise SO2NBTrainingError("best_epoch cannot exceed completed_epoch.")
        if self.best_value is None and self.best_epoch != 0:
            raise SO2NBTrainingError("An unset best value requires best_epoch=0.")
        if self.stop_reason not in {None, "patience", "maximum_epochs"}:
            raise SO2NBTrainingError("Unknown early-stopping reason.")


def update_early_stopping(
    state: EarlyStoppingState,
    validation_value: float,
    *,
    completed_epoch: int,
    minimum_epochs: int = 50,
    patience: int = 25,
    min_delta: float = 1e-4,
    maximum_epochs: int = 300,
) -> EarlyStoppingState:
    """Advance the frozen stop rule by exactly one completed validation.

    Best-model tracking starts at epoch one.  Patience starts only *after* the
    first ``minimum_epochs`` completed epochs, so the earliest patience stop is
    epoch 75 for the frozen 50/25 settings.
    """

    if state.should_stop:
        raise SO2NBTrainingError("Cannot advance an already stopped state.")
    epoch = int(completed_epoch)
    if epoch != state.completed_epoch + 1:
        raise SO2NBTrainingError("Early-stopping epochs must be contiguous.")
    if minimum_epochs < 0 or patience <= 0 or maximum_epochs <= 0:
        raise SO2NBTrainingError("Stopping bounds are invalid.")
    if minimum_epochs >= maximum_epochs:
        raise SO2NBTrainingError("minimum_epochs must be below maximum_epochs.")
    delta = _finite(min_delta, field="min_delta", nonnegative=True)
    value = _finite(validation_value, field="validation_value")

    improved = state.best_value is None or value < state.best_value - delta
    if improved:
        best_value = value
        best_epoch = epoch
        bad = 0
    else:
        best_value = state.best_value
        best_epoch = state.best_epoch
        bad = state.bad_validations + (1 if epoch > minimum_epochs else 0)

    stop_reason: str | None = None
    if epoch >= maximum_epochs:
        stop_reason = "maximum_epochs"
    elif epoch > minimum_epochs and bad >= patience:
        stop_reason = "patience"
    return EarlyStoppingState(
        best_value=best_value,
        best_epoch=best_epoch,
        bad_validations=bad,
        completed_epoch=epoch,
        improved=improved,
        should_stop=stop_reason is not None,
        stop_reason=stop_reason,
    )


@dataclass(frozen=True)
class ValidationViewStatistics:
    """Additive sufficient statistics for one fixed core/mask view."""

    alias: str
    view_index: int
    n_masked_entries: int
    negative_binomial_nll_sum: float
    raw_absolute_error_sum: float
    raw_squared_error_sum: float
    log1p_absolute_error_sum: float
    log1p_squared_error_sum: float
    poisson_deviance_sum: float
    observed_zero_sum: float
    predicted_zero_probability_sum: float
    zero_brier_sum: float
    observed_count_sum: float
    predicted_count_sum: float

    def __post_init__(self) -> None:
        alias = str(self.alias).strip().upper()
        if not alias:
            raise SO2NBTrainingError("Validation alias cannot be empty.")
        if int(self.view_index) < 0:
            raise SO2NBTrainingError("Validation view index cannot be negative.")
        if int(self.n_masked_entries) <= 0:
            raise SO2NBTrainingError("A validation view must mask entries.")
        for name in (
            "negative_binomial_nll_sum",
            "raw_absolute_error_sum",
            "raw_squared_error_sum",
            "log1p_absolute_error_sum",
            "log1p_squared_error_sum",
            "poisson_deviance_sum",
            "observed_zero_sum",
            "predicted_zero_probability_sum",
            "zero_brier_sum",
            "observed_count_sum",
            "predicted_count_sum",
        ):
            _finite(getattr(self, name), field=name, nonnegative=True)
        n = int(self.n_masked_entries)
        if self.observed_zero_sum > n + 1e-6:
            raise SO2NBTrainingError("Observed-zero sum exceeds the masked count.")
        if self.predicted_zero_probability_sum > n + 1e-6:
            raise SO2NBTrainingError("Predicted-zero sum exceeds the masked count.")
        if self.zero_brier_sum > n + 1e-6:
            raise SO2NBTrainingError("Zero-Brier sum exceeds the masked count.")
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "view_index", int(self.view_index))
        object.__setattr__(self, "n_masked_entries", n)

    @classmethod
    def from_metrics(
        cls,
        *,
        alias: str,
        view_index: int,
        metrics: Any,
        observed_count_sum: float,
        predicted_count_sum: float,
    ) -> "ValidationViewStatistics":
        """Convert the public NB metric object into additive statistics."""

        n = int(metrics.n_masked_entries)
        return cls(
            alias=alias,
            view_index=view_index,
            n_masked_entries=n,
            negative_binomial_nll_sum=float(metrics.negative_binomial_nll) * n,
            raw_absolute_error_sum=float(metrics.raw_count_mae) * n,
            raw_squared_error_sum=float(metrics.raw_count_rmse) ** 2 * n,
            log1p_absolute_error_sum=float(metrics.log1p_mae) * n,
            log1p_squared_error_sum=float(metrics.log1p_rmse) ** 2 * n,
            poisson_deviance_sum=float(metrics.poisson_deviance) * n,
            observed_zero_sum=float(metrics.observed_zero_rate) * n,
            predicted_zero_probability_sum=(
                float(metrics.predicted_zero_probability_mean) * n
            ),
            zero_brier_sum=float(metrics.zero_brier_score) * n,
            observed_count_sum=observed_count_sum,
            predicted_count_sum=predicted_count_sum,
        )


_VALIDATION_TENSOR_FIELDS = (
    "n_masked_entries",
    "negative_binomial_nll_sum",
    "raw_absolute_error_sum",
    "raw_squared_error_sum",
    "log1p_absolute_error_sum",
    "log1p_squared_error_sum",
    "poisson_deviance_sum",
    "observed_zero_sum",
    "predicted_zero_probability_sum",
    "zero_brier_sum",
    "observed_count_sum",
    "predicted_count_sum",
    "present",
)


def validation_statistics_tensor(
    records: Sequence[ValidationViewStatistics],
    *,
    aliases: Sequence[str] = VALIDATION_ALIASES,
    views_per_core: int = VALIDATION_VIEWS_PER_CORE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode rank-local validation records for one exact DDP SUM reduction."""

    canonical = tuple(str(alias).strip().upper() for alias in aliases)
    alias_index = {alias: index for index, alias in enumerate(canonical)}
    tensor = torch.zeros(
        (len(canonical), int(views_per_core), len(_VALIDATION_TENSOR_FIELDS)),
        dtype=torch.float64,
        device=device,
    )
    seen: set[tuple[str, int]] = set()
    for record in records:
        key = (record.alias, record.view_index)
        if record.alias not in alias_index:
            raise SO2NBTrainingError(f"Unexpected validation alias {record.alias}.")
        if record.view_index not in range(int(views_per_core)):
            raise SO2NBTrainingError("Validation view index is out of range.")
        if key in seen:
            raise SO2NBTrainingError("A rank supplied a duplicate validation view.")
        seen.add(key)
        values = [
            float(record.n_masked_entries),
            float(record.negative_binomial_nll_sum),
            float(record.raw_absolute_error_sum),
            float(record.raw_squared_error_sum),
            float(record.log1p_absolute_error_sum),
            float(record.log1p_squared_error_sum),
            float(record.poisson_deviance_sum),
            float(record.observed_zero_sum),
            float(record.predicted_zero_probability_sum),
            float(record.zero_brier_sum),
            float(record.observed_count_sum),
            float(record.predicted_count_sum),
            1.0,
        ]
        tensor[alias_index[record.alias], record.view_index] = torch.tensor(
            values, dtype=torch.float64, device=tensor.device
        )
    return tensor


def validation_statistics_from_tensor(
    tensor: torch.Tensor,
    *,
    aliases: Sequence[str] = VALIDATION_ALIASES,
    views_per_core: int = VALIDATION_VIEWS_PER_CORE,
) -> tuple[ValidationViewStatistics, ...]:
    """Decode a globally reduced tensor and require every view exactly once."""

    canonical = tuple(str(alias).strip().upper() for alias in aliases)
    expected_shape = (
        len(canonical),
        int(views_per_core),
        len(_VALIDATION_TENSOR_FIELDS),
    )
    if tuple(tensor.shape) != expected_shape:
        raise SO2NBTrainingError(
            f"Validation tensor shape {tuple(tensor.shape)} != {expected_shape}."
        )
    values = tensor.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise SO2NBTrainingError("Reduced validation statistics are non-finite.")
    if not bool(values[..., -1].eq(1.0).all()):
        raise SO2NBTrainingError(
            "Every fixed validation core/view must be evaluated exactly once."
        )
    records: list[ValidationViewStatistics] = []
    for alias_index, alias in enumerate(canonical):
        for view_index in range(int(views_per_core)):
            row = values[alias_index, view_index]
            records.append(
                ValidationViewStatistics(
                    alias=alias,
                    view_index=view_index,
                    n_masked_entries=int(row[0].item()),
                    negative_binomial_nll_sum=float(row[1].item()),
                    raw_absolute_error_sum=float(row[2].item()),
                    raw_squared_error_sum=float(row[3].item()),
                    log1p_absolute_error_sum=float(row[4].item()),
                    log1p_squared_error_sum=float(row[5].item()),
                    poisson_deviance_sum=float(row[6].item()),
                    observed_zero_sum=float(row[7].item()),
                    predicted_zero_probability_sum=float(row[8].item()),
                    zero_brier_sum=float(row[9].item()),
                    observed_count_sum=float(row[10].item()),
                    predicted_count_sum=float(row[11].item()),
                )
            )
    return tuple(records)


@dataclass(frozen=True)
class ValidationAggregate:
    primary_equal_core_nll: float
    pooled_nll: float
    raw_count_mae: float
    raw_count_rmse: float
    log1p_mae: float
    log1p_rmse: float
    poisson_deviance: float
    observed_zero_rate: float
    predicted_zero_probability_mean: float
    zero_brier_score: float
    total_masked_entries: int
    per_core: Mapping[str, Mapping[str, float | int]]
    per_view: Mapping[str, Mapping[str, float | int]]


def aggregate_validation_statistics(
    records: Sequence[ValidationViewStatistics],
    *,
    aliases: Sequence[str] = VALIDATION_ALIASES,
    views_per_core: int = VALIDATION_VIEWS_PER_CORE,
) -> ValidationAggregate:
    """Apply mean-view-within-core, then equal-core primary aggregation."""

    canonical = tuple(str(alias).strip().upper() for alias in aliases)
    by_key: dict[tuple[str, int], ValidationViewStatistics] = {}
    for record in records:
        key = (record.alias, record.view_index)
        if key in by_key:
            raise SO2NBTrainingError("Duplicate validation core/view statistics.")
        by_key[key] = record
    expected = {
        (alias, view_index)
        for alias in canonical
        for view_index in range(int(views_per_core))
    }
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise SO2NBTrainingError(
            f"Validation view coverage mismatch; missing={missing}, extra={extra}."
        )

    per_core: dict[str, Mapping[str, float | int]] = {}
    per_view: dict[str, Mapping[str, float | int]] = {}
    core_primary: list[float] = []
    totals = np.zeros(10, dtype=np.float64)
    for alias in canonical:
        views = [by_key[(alias, index)] for index in range(int(views_per_core))]
        for item in views:
            key = f"{alias}:{item.view_index}"
            per_view[key] = {
                "core_alias": alias,
                "view_index": item.view_index,
                "masked_entries": item.n_masked_entries,
                "masked_negative_binomial_nll": (
                    item.negative_binomial_nll_sum / item.n_masked_entries
                ),
                "observed_count_mean": (
                    item.observed_count_sum / item.n_masked_entries
                ),
                "predicted_count_mean": (
                    item.predicted_count_sum / item.n_masked_entries
                ),
            }
        nll_by_view = [
            item.negative_binomial_nll_sum / item.n_masked_entries for item in views
        ]
        core_n = sum(item.n_masked_entries for item in views)
        core_nll_sum = sum(item.negative_binomial_nll_sum for item in views)
        core_raw_absolute_sum = sum(item.raw_absolute_error_sum for item in views)
        core_raw_squared_sum = sum(item.raw_squared_error_sum for item in views)
        core_log_absolute_sum = sum(item.log1p_absolute_error_sum for item in views)
        core_log_squared_sum = sum(item.log1p_squared_error_sum for item in views)
        core_poisson_sum = sum(item.poisson_deviance_sum for item in views)
        core_observed_zero_sum = sum(item.observed_zero_sum for item in views)
        core_predicted_zero_sum = sum(
            item.predicted_zero_probability_sum for item in views
        )
        core_zero_brier_sum = sum(item.zero_brier_sum for item in views)
        core_primary_value = float(np.mean(nll_by_view, dtype=np.float64))
        core_primary.append(core_primary_value)
        per_core[alias] = {
            "masked_negative_binomial_nll": core_primary_value,
            "pooled_masked_negative_binomial_nll": core_nll_sum / core_n,
            "masked_entries": core_n,
            "masked_raw_count_mae": core_raw_absolute_sum / core_n,
            "masked_raw_count_rmse": math.sqrt(core_raw_squared_sum / core_n),
            "masked_log1p_mae": core_log_absolute_sum / core_n,
            "masked_log1p_rmse": math.sqrt(core_log_squared_sum / core_n),
            "masked_poisson_deviance": core_poisson_sum / core_n,
            "observed_zero_rate": core_observed_zero_sum / core_n,
            "predicted_zero_probability_mean": core_predicted_zero_sum / core_n,
            "zero_brier_score": core_zero_brier_sum / core_n,
        }
        totals += np.asarray(
            [
                core_n,
                core_nll_sum,
                core_raw_absolute_sum,
                core_raw_squared_sum,
                core_log_absolute_sum,
                core_log_squared_sum,
                core_poisson_sum,
                core_observed_zero_sum,
                core_predicted_zero_sum,
                core_zero_brier_sum,
            ],
            dtype=np.float64,
        )
    n = int(totals[0])
    if n <= 0:
        raise SO2NBTrainingError("Validation aggregation has no masked entries.")
    result = ValidationAggregate(
        primary_equal_core_nll=float(np.mean(core_primary, dtype=np.float64)),
        pooled_nll=float(totals[1] / n),
        raw_count_mae=float(totals[2] / n),
        raw_count_rmse=float(math.sqrt(totals[3] / n)),
        log1p_mae=float(totals[4] / n),
        log1p_rmse=float(math.sqrt(totals[5] / n)),
        poisson_deviance=float(totals[6] / n),
        observed_zero_rate=float(totals[7] / n),
        predicted_zero_probability_mean=float(totals[8] / n),
        zero_brier_score=float(totals[9] / n),
        total_masked_entries=n,
        per_core=per_core,
        per_view=per_view,
    )
    for name, value in asdict(result).items():
        if name not in {"per_core", "per_view", "total_masked_entries"}:
            _finite(value, field=f"validation.{name}", nonnegative=True)
    for scope, records_by_key in (("per_core", per_core), ("per_view", per_view)):
        for key, metrics in records_by_key.items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    _finite(
                        value,
                        field=f"validation.{scope}.{key}.{metric}",
                        nonnegative=True,
                    )
    return result


def aggregate_training_nll(
    view_losses: Sequence[Mapping[str, Any]],
    *,
    aliases: Sequence[str] = TRAINING_ALIASES,
    views_per_core: int = 10,
) -> tuple[float, Mapping[str, float]]:
    """Return the exact equal-view-within-core, then equal-core train NLL."""

    canonical = tuple(str(alias).strip().upper() for alias in aliases)
    indexed: dict[tuple[str, int], float] = {}
    for record in view_losses:
        alias = str(record["alias"]).strip().upper()
        view = int(record["view_index"])
        key = (alias, view)
        if key in indexed:
            raise SO2NBTrainingError("Duplicate training core/view loss.")
        indexed[key] = _finite(record["negative_binomial_nll"], field="train_nll")
    expected = {
        (alias, view)
        for alias in canonical
        for view in range(int(views_per_core))
    }
    if set(indexed) != expected:
        raise SO2NBTrainingError("Training losses do not cover every core/view.")
    per_core = {
        alias: float(
            np.mean(
                [indexed[(alias, view)] for view in range(int(views_per_core))],
                dtype=np.float64,
            )
        )
        for alias in canonical
    }
    return float(np.mean(list(per_core.values()), dtype=np.float64)), per_core


def capture_rng_state() -> Mapping[str, Any]:
    """Capture this rank's Python/NumPy/Torch RNG state at an epoch boundary."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise SO2NBTrainingError("Checkpoint requires a CUDA RNG state.")
        torch.cuda.set_rng_state(torch.as_tensor(cuda_state, dtype=torch.uint8).cpu())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class CheckpointReceipt:
    path: Path
    sha256: str
    completed_epoch: int
    role: str


class AtomicBestLatestCheckpointStore:
    """Retain at most one recovery and one validation-best checkpoint."""

    def __init__(self, run_root: str | Path, *, rank: int = 0) -> None:
        if int(rank) != 0:
            raise SO2NBTrainingError("Only rank zero may write checkpoints.")
        self.directory = Path(run_root) / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.ckpt"
        self.best_path = self.directory / "best.ckpt"
        unexpected = sorted(
            path.name
            for path in self.directory.iterdir()
            if path.is_file() and path.name not in {"latest.ckpt", "best.ckpt"}
        )
        if unexpected:
            raise SO2NBTrainingError(
                "Checkpoint directory contains epoch archives or unknown files: "
                + ", ".join(unexpected)
            )

    @staticmethod
    def _validate_payload(payload: Mapping[str, Any], *, role: str) -> int:
        if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
            raise SO2NBTrainingError("Checkpoint schema mismatch.")
        if payload.get("protocol") != PROTOCOL:
            raise SO2NBTrainingError("Checkpoint protocol mismatch.")
        if payload.get("checkpoint_role") != role:
            raise SO2NBTrainingError("Checkpoint role mismatch.")
        epoch = int(payload.get("completed_epoch", -1))
        if epoch <= 0:
            raise SO2NBTrainingError("Checkpoint has no completed epoch.")
        early = payload.get("early_stopping_state")
        if not isinstance(early, Mapping):
            raise SO2NBTrainingError("Checkpoint lacks early-stopping state.")
        early_state = EarlyStoppingState(**dict(early))
        if early_state.completed_epoch != epoch:
            raise SO2NBTrainingError(
                "Checkpoint epoch differs from its early-stopping epoch."
            )
        for field in ("preprocessing_fingerprint", "split_fingerprint"):
            if not _is_sha256(payload.get(field)):
                raise SO2NBTrainingError(f"Checkpoint {field} is invalid.")
        if role == "best":
            if early_state.best_epoch != epoch:
                raise SO2NBTrainingError("Best checkpoint epoch is not its best epoch.")
            if payload.get("embedded_best_checkpoint") is not None:
                raise SO2NBTrainingError("A best checkpoint cannot recursively embed itself.")
            if payload.get("best_checkpoint_sha256") is not None:
                raise SO2NBTrainingError("A best checkpoint cannot reference a sibling best.")
        else:
            embedded = payload.get("embedded_best_checkpoint")
            if not isinstance(embedded, Mapping):
                raise SO2NBTrainingError(
                    "Latest checkpoint lacks its transactionally embedded best state."
                )
            AtomicBestLatestCheckpointStore._validate_payload(embedded, role="best")
            if not _is_sha256(payload.get("best_checkpoint_sha256")):
                raise SO2NBTrainingError("Latest checkpoint's best checksum is invalid.")
            if not _is_sha256(payload.get("embedded_best_checkpoint_tree_sha256")):
                raise SO2NBTrainingError("Embedded best tree checksum is invalid.")
            if int(embedded.get("completed_epoch", -1)) != int(
                early.get("best_epoch", -2)
            ):
                raise SO2NBTrainingError(
                    "Embedded best epoch does not match latest early-stopping state."
                )
            if embedded.get("best_validation_metric") != early.get("best_value"):
                raise SO2NBTrainingError(
                    "Embedded best metric does not match latest early-stopping state."
                )
        return epoch

    def _write(self, payload: Mapping[str, Any], *, role: str) -> CheckpointReceipt:
        epoch = self._validate_payload(payload, role=role)
        destination = self.latest_path if role == "latest" else self.best_path
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".writing", dir=self.directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(dict(payload), temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            loaded = torch.load(temporary, map_location="cpu", weights_only=False)
            if not isinstance(loaded, Mapping):
                raise SO2NBTrainingError("Serialized checkpoint is not a mapping.")
            self._validate_payload(loaded, role=role)
            digest = sha256_file(temporary)
            os.replace(temporary, destination)
            _fsync_directory(self.directory)
        finally:
            temporary.unlink(missing_ok=True)
        return CheckpointReceipt(destination, digest, epoch, role)

    def save_latest(self, payload: Mapping[str, Any]) -> CheckpointReceipt:
        return self._write(payload, role="latest")

    def save_best(self, payload: Mapping[str, Any]) -> CheckpointReceipt:
        return self._write(payload, role="best")

    def load(self, role: str) -> Mapping[str, Any]:
        if role not in {"latest", "best"}:
            raise SO2NBTrainingError("Checkpoint role must be latest or best.")
        path = self.latest_path if role == "latest" else self.best_path
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise SO2NBTrainingError("Checkpoint payload is not a mapping.")
        self._validate_payload(payload, role=role)
        return payload

    def discard_best_without_latest(self) -> bool:
        """Remove an uncommitted best left before the first latest commit."""

        if self.latest_path.exists() or self.latest_path.is_symlink():
            raise SO2NBTrainingError(
                "Cannot discard best while a committed latest checkpoint exists."
            )
        existed = self.best_path.exists() or self.best_path.is_symlink()
        self.best_path.unlink(missing_ok=True)
        if existed:
            _fsync_directory(self.directory)
        return existed

    def finalize_best_only(self) -> CheckpointReceipt:
        """Verify best is loadable, then remove recovery state and nothing else."""

        payload = self.load("best")
        epoch = self._validate_payload(payload, role="best")
        digest = sha256_file(self.best_path)
        self.latest_path.unlink(missing_ok=True)
        _fsync_directory(self.directory)
        remaining = sorted(path.name for path in self.directory.iterdir() if path.is_file())
        if remaining != ["best.ckpt"]:
            raise SO2NBTrainingError(
                f"Final checkpoint layout must be best-only; found {remaining}."
            )
        return CheckpointReceipt(self.best_path, digest, epoch, "best")


EPOCH_METRIC_COLUMNS = (
    "schema",
    "global_epoch",
    "training_equal_core_masked_nb_nll",
    "validation_equal_core_masked_nb_nll",
    "validation_pooled_masked_nb_nll",
    "validation_masked_raw_count_mae",
    "validation_masked_raw_count_rmse",
    "validation_masked_log1p_mae",
    "validation_masked_log1p_rmse",
    "validation_masked_poisson_deviance",
    "validation_observed_zero_rate",
    "validation_predicted_zero_probability_mean",
    "validation_zero_brier_score",
    "inverse_dispersion_min",
    "inverse_dispersion_median",
    "inverse_dispersion_max",
    "learning_rate",
    "epoch_duration_seconds",
    "training_views_per_second",
    "peak_vram_gib_all_ranks",
    "best_validation_metric",
    "best_epoch",
    "bad_validations",
    "improved",
    "should_stop",
    "stop_reason",
    "global_gradient_json",
    "block_gradients_json",
    "dispersion_gradient_json",
    "training_per_core_json",
    "validation_per_core_json",
    "training_mask_receipts_json",
    "validation_mask_checksums_json",
)


class DurableNBEpochMetricsCSV:
    """Atomically publish one complete, contiguous scalar row per epoch."""

    def __init__(self, run_root: str | Path, *, rank: int = 0) -> None:
        if int(rank) != 0:
            raise SO2NBTrainingError("Only rank zero may write epoch metrics.")
        self.path = Path(run_root) / "results" / "epoch_metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and tuple(rows[0]) != EPOCH_METRIC_COLUMNS:
            raise SO2NBTrainingError("Epoch-metric CSV schema drifted.")
        epochs = [int(row["global_epoch"]) for row in rows]
        if epochs != list(range(1, len(rows) + 1)):
            raise SO2NBTrainingError("Epoch-metric rows must be unique and contiguous.")
        return rows

    def _rewrite(self, rows: Sequence[Mapping[str, Any]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".epoch_metrics.csv.", suffix=".writing", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=EPOCH_METRIC_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def reconcile(self, *, checkpoint_epoch: int) -> int:
        rows = self.rows()
        epoch = int(checkpoint_epoch)
        if epoch < 0 or len(rows) < epoch:
            raise SO2NBTrainingError("Metrics cannot reconcile to the checkpoint epoch.")
        if len(rows) > epoch:
            self._rewrite(rows[:epoch])
        return epoch

    def append(self, row: Mapping[str, Any]) -> None:
        if set(row) != set(EPOCH_METRIC_COLUMNS):
            missing = sorted(set(EPOCH_METRIC_COLUMNS) - set(row))
            extra = sorted(set(row) - set(EPOCH_METRIC_COLUMNS))
            raise SO2NBTrainingError(
                f"Epoch metric columns mismatch; missing={missing}, extra={extra}."
            )
        normalized = dict(row)
        normalized["schema"] = EPOCH_METRICS_SCHEMA
        epoch = int(normalized["global_epoch"])
        rows = self.rows()
        if epoch != len(rows) + 1:
            raise SO2NBTrainingError("Epoch metrics must append contiguously.")
        for key, value in normalized.items():
            if key.endswith("_json"):
                if not isinstance(value, str):
                    normalized[key] = json.dumps(
                        value, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
        self._rewrite([*rows, normalized])


PER_CORE_METRIC_COLUMNS = (
    "schema",
    "global_epoch",
    "split",
    "core_alias",
    "mask_views",
    "masked_entries",
    "masked_negative_binomial_nll",
    "pooled_masked_negative_binomial_nll",
    "masked_raw_count_mae",
    "masked_raw_count_rmse",
    "masked_log1p_mae",
    "masked_log1p_rmse",
    "masked_poisson_deviance",
    "observed_zero_rate",
    "predicted_zero_probability_mean",
    "zero_brier_score",
)


class DurableNBPerCoreMetricsCSV:
    """Persist exactly twelve train and two validation core rows per epoch."""

    def __init__(self, run_root: str | Path, *, rank: int = 0) -> None:
        if int(rank) != 0:
            raise SO2NBTrainingError("Only rank zero may write per-core metrics.")
        self.path = Path(run_root) / "results" / "per_core_epoch_metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and tuple(rows[0]) != PER_CORE_METRIC_COLUMNS:
            raise SO2NBTrainingError("Per-core metric CSV schema drifted.")
        return rows

    def _rewrite(self, rows: Sequence[Mapping[str, Any]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".per_core_epoch_metrics.csv.",
            suffix=".writing",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PER_CORE_METRIC_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def reconcile(self, *, checkpoint_epoch: int) -> None:
        epoch = int(checkpoint_epoch)
        rows = self.rows()
        retained = [row for row in rows if int(row["global_epoch"]) <= epoch]
        observed = {
            int(row["global_epoch"]): sum(
                1 for item in retained if int(item["global_epoch"]) == int(row["global_epoch"])
            )
            for row in retained
        }
        if any(count != 14 for count in observed.values()) or set(observed) != set(
            range(1, epoch + 1)
        ):
            raise SO2NBTrainingError("Per-core rows do not cover complete epochs.")
        if len(retained) != len(rows):
            self._rewrite(retained)

    def append_epoch(
        self,
        *,
        global_epoch: int,
        training_per_core: Mapping[str, float],
        validation_per_core: Mapping[str, Mapping[str, float | int]],
    ) -> None:
        epoch = int(global_epoch)
        if tuple(training_per_core) != TRAINING_ALIASES:
            raise SO2NBTrainingError("Training per-core metrics are not canonically ordered.")
        if tuple(validation_per_core) != VALIDATION_ALIASES:
            raise SO2NBTrainingError("Validation per-core metrics are not canonically ordered.")
        existing = self.rows()
        expected_epoch = len(existing) // 14 + 1
        if len(existing) % 14 or epoch != expected_epoch:
            raise SO2NBTrainingError("Per-core metrics must append one complete epoch.")
        appended: list[dict[str, Any]] = []
        for alias, nll in training_per_core.items():
            appended.append(
                {
                    "schema": EPOCH_METRICS_SCHEMA,
                    "global_epoch": epoch,
                    "split": "training",
                    "core_alias": alias,
                    "mask_views": 10,
                    "masked_entries": "",
                    "masked_negative_binomial_nll": _finite(
                        nll, field=f"training.{alias}.nll", nonnegative=True
                    ),
                    "pooled_masked_negative_binomial_nll": "",
                    "masked_raw_count_mae": "",
                    "masked_raw_count_rmse": "",
                    "masked_log1p_mae": "",
                    "masked_log1p_rmse": "",
                    "masked_poisson_deviance": "",
                    "observed_zero_rate": "",
                    "predicted_zero_probability_mean": "",
                    "zero_brier_score": "",
                }
            )
        for alias, metrics in validation_per_core.items():
            appended.append(
                {
                    "schema": EPOCH_METRICS_SCHEMA,
                    "global_epoch": epoch,
                    "split": "validation",
                    "core_alias": alias,
                    "mask_views": VALIDATION_VIEWS_PER_CORE,
                    "masked_entries": int(metrics["masked_entries"]),
                    "masked_negative_binomial_nll": _finite(
                        metrics["masked_negative_binomial_nll"],
                        field=f"validation.{alias}.nll",
                        nonnegative=True,
                    ),
                    "pooled_masked_negative_binomial_nll": _finite(
                        metrics["pooled_masked_negative_binomial_nll"],
                        field=f"validation.{alias}.pooled_nll",
                        nonnegative=True,
                    ),
                    "masked_raw_count_mae": _finite(
                        metrics["masked_raw_count_mae"],
                        field=f"validation.{alias}.raw_mae",
                        nonnegative=True,
                    ),
                    "masked_raw_count_rmse": _finite(
                        metrics["masked_raw_count_rmse"],
                        field=f"validation.{alias}.raw_rmse",
                        nonnegative=True,
                    ),
                    "masked_log1p_mae": _finite(
                        metrics["masked_log1p_mae"],
                        field=f"validation.{alias}.log_mae",
                        nonnegative=True,
                    ),
                    "masked_log1p_rmse": _finite(
                        metrics["masked_log1p_rmse"],
                        field=f"validation.{alias}.log_rmse",
                        nonnegative=True,
                    ),
                    "masked_poisson_deviance": _finite(
                        metrics["masked_poisson_deviance"],
                        field=f"validation.{alias}.poisson_deviance",
                        nonnegative=True,
                    ),
                    "observed_zero_rate": _finite(
                        metrics["observed_zero_rate"],
                        field=f"validation.{alias}.observed_zero_rate",
                        nonnegative=True,
                    ),
                    "predicted_zero_probability_mean": _finite(
                        metrics["predicted_zero_probability_mean"],
                        field=f"validation.{alias}.predicted_zero_probability_mean",
                        nonnegative=True,
                    ),
                    "zero_brier_score": _finite(
                        metrics["zero_brier_score"],
                        field=f"validation.{alias}.zero_brier_score",
                        nonnegative=True,
                    ),
                }
            )
        self._rewrite([*existing, *appended])


class DurableScalarCSV:
    """Small atomic CSV used for scalar-only gradient summaries."""

    def __init__(
        self,
        run_root: str | Path,
        relative_path: str | Path,
        columns: Sequence[str],
        *,
        rows_per_epoch: int,
        rank: int = 0,
    ) -> None:
        if int(rank) != 0:
            raise SO2NBTrainingError("Only rank zero may write gradient metrics.")
        self.path = Path(run_root) / relative_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = tuple(columns)
        self.rows_per_epoch = int(rows_per_epoch)
        if not self.columns or self.rows_per_epoch <= 0 or "global_epoch" not in self.columns:
            raise SO2NBTrainingError("Scalar CSV contract is invalid.")

    def rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and tuple(rows[0]) != self.columns:
            raise SO2NBTrainingError("Scalar CSV schema drifted.")
        return rows

    def _rewrite(self, rows: Sequence[Mapping[str, Any]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".writing", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def reconcile(self, *, checkpoint_epoch: int) -> None:
        epoch = int(checkpoint_epoch)
        rows = self.rows()
        retained = [row for row in rows if int(row["global_epoch"]) <= epoch]
        if len(retained) != epoch * self.rows_per_epoch:
            raise SO2NBTrainingError("Scalar summaries do not cover checkpoint epochs.")
        if len(retained) != len(rows):
            self._rewrite(retained)

    def append_epoch(self, rows: Sequence[Mapping[str, Any]], *, global_epoch: int) -> None:
        materialized = [dict(row) for row in rows]
        if len(materialized) != self.rows_per_epoch:
            raise SO2NBTrainingError("Wrong scalar-summary row count for one epoch.")
        if any(set(row) != set(self.columns) for row in materialized):
            raise SO2NBTrainingError("Scalar-summary columns mismatch.")
        epoch = int(global_epoch)
        if any(int(row["global_epoch"]) != epoch for row in materialized):
            raise SO2NBTrainingError("Scalar-summary epoch mismatch.")
        existing = self.rows()
        if len(existing) != (epoch - 1) * self.rows_per_epoch:
            raise SO2NBTrainingError("Scalar summaries must append contiguously.")
        self._rewrite([*existing, *materialized])


def write_nb_training_plots(
    run_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, str]:
    """Write compact loss and gradient-direction figures from scalar rows."""

    if not rows:
        raise SO2NBTrainingError("Cannot plot an empty epoch history.")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root = Path(run_root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray([int(row["global_epoch"]) for row in rows])
    train = np.asarray(
        [float(row["training_equal_core_masked_nb_nll"]) for row in rows]
    )
    validation = np.asarray(
        [float(row["validation_equal_core_masked_nb_nll"]) for row in rows]
    )
    best_epoch = int(rows[-1]["best_epoch"])

    loss_path = figures / "loss_vs_epoch.png"
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train, label="training NB NLL", linewidth=1.4)
    axis.plot(epochs, validation, label="validation NB NLL", linewidth=1.4)
    axis.axvline(best_epoch, color="black", linestyle="--", linewidth=0.9, label="best")
    axis.set(xlabel="global epoch", ylabel="masked NB NLL")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    gradient_path = figures / "gradient_direction_vs_epoch.png"
    fig, axis = plt.subplots(figsize=(8, 5))
    for field, label in (
        ("global_gradient_json", "global"),
        ("dispersion_gradient_json", "inverse dispersion"),
    ):
        summaries = [
            value if isinstance(value := row[field], Mapping) else json.loads(value)
            for row in rows
        ]
        values = [
            summary.get("epoch_aggregate_gradient_cosine_to_previous_epoch")
            for summary in summaries
        ]
        axis.plot(
            epochs,
            [np.nan if value is None else float(value) for value in values],
            label=label,
        )
    block_summaries = [
        value if isinstance(value := row["block_gradients_json"], Sequence)
        and not isinstance(value, (str, bytes))
        else json.loads(value)
        for row in rows
    ]
    for block_index in range(4):
        values: list[float] = []
        for summaries in block_summaries:
            by_index = {
                int(summary["block_index"]): summary
                for summary in summaries
                if isinstance(summary, Mapping)
            }
            if set(by_index) != set(range(4)):
                raise SO2NBTrainingError(
                    "Every epoch must contain four ordered block-gradient summaries."
                )
            value = by_index[block_index].get(
                "epoch_aggregate_gradient_cosine_to_previous_epoch"
            )
            values.append(np.nan if value is None else float(value))
        axis.plot(epochs, values, label=f"block {block_index}", linewidth=1.0)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set(xlabel="global epoch", ylabel="gradient cosine to previous epoch", ylim=(-1.05, 1.05))
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(gradient_path, dpi=160)
    plt.close(fig)
    return {
        "loss_vs_epoch": "figures/loss_vs_epoch.png",
        "gradient_direction_vs_epoch": "figures/gradient_direction_vs_epoch.png",
    }


__all__ = [
    "AtomicBestLatestCheckpointStore",
    "CHECKPOINT_SCHEMA",
    "CheckpointReceipt",
    "DurableNBPerCoreMetricsCSV",
    "DurableNBEpochMetricsCSV",
    "DurableScalarCSV",
    "EPOCH_METRICS_SCHEMA",
    "EPOCH_METRIC_COLUMNS",
    "EXPECTED_PARAMETER_COUNT",
    "EarlyStoppingState",
    "OPTIMIZER_UPDATES_PER_EPOCH",
    "PER_CORE_METRIC_COLUMNS",
    "PROTOCOL",
    "SO2NBTrainingError",
    "TRAINING_ALIASES",
    "VALIDATION_ALIASES",
    "VALIDATION_VIEWS_PER_CORE",
    "ValidationAggregate",
    "ValidationViewStatistics",
    "aggregate_training_nll",
    "aggregate_validation_statistics",
    "capture_rng_state",
    "restore_rng_state",
    "sha256_file",
    "update_early_stopping",
    "validation_statistics_from_tensor",
    "validation_statistics_tensor",
    "write_nb_training_plots",
]

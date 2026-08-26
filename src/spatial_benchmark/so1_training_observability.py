"""Durable live metrics and latest-only checkpoints for SO1 training.

The helpers in this module deliberately do not own optimization.  The cohort
trainer calls them only after a complete global epoch.  This keeps wall-clock
observability outside deterministic training history while giving an operator
one CSV that is safe to download during a run.

Only rank zero may construct either writer.  A rolling checkpoint is always
named ``latest.ckpt`` and is atomically replaced; successful completion
atomically renames that file to ``last.ckpt``.  Consequently there is never a
collection of periodic model files to prune.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .pooled_relative_qkv_training import (
    CHECKPOINT_INTERVAL_GLOBAL_EPOCHS,
    PLATEAU_FIRST_ALLOWED_STOP_EPOCH,
    PLATEAU_WINDOW_GLOBAL_EPOCHS,
    audit_training_loss_plateau,
    single_seed_plateau_decision,
)


SO1_CORE_ALIASES = tuple(f"SO1-C{core:02d}" for core in range(1, 15))
EPOCH_METRICS_SCHEMA = "so1_14core_epoch_metrics_v1"
CHECKPOINT_SCHEMA = "so1_14core_relative_qkv_resume_v1"
STRICT_PLATEAU_ABSOLUTE_RELATIVE_HALF_WINDOW_CHANGE_MAX = 0.0005
STRICT_PLATEAU_NORMALIZED_ABSOLUTE_SLOPE_MAX = 0.000025
STRICT_PLATEAU_CONSECUTIVE_PASSING_AUDITS = 2


class SO1ObservabilityError(RuntimeError):
    """Raised when live metrics or checkpoint durability would be ambiguous."""


def _loss_column(alias: str) -> str:
    return "loss_" + alias.lower().replace("-", "_")


CORE_LOSS_COLUMNS = tuple(_loss_column(alias) for alias in SO1_CORE_ALIASES)
EPOCH_METRICS_COLUMNS = (
    "schema",
    "run_id",
    "model_seed",
    "global_epoch",
    "equal_core_mean_masked_huber",
    *CORE_LOSS_COLUMNS,
    "gradient_norm_mean_before_clip",
    "gradient_norm_max_before_clip",
    "learning_rate",
    "optimizer_updates_this_epoch",
    "cumulative_optimizer_updates",
    "epoch_duration_seconds",
    "rolling_5_epoch_duration_seconds",
    "elapsed_training_seconds",
    "eta_to_next_plateau_audit_seconds",
    "eta_to_epoch_200_seconds",
    "complete_graph_mask_views",
    "masked_entries_across_views",
    "complete_graph_views_per_second",
    "masked_entries_per_second",
    "peak_vram_gib_all_ranks",
    "plateau_audit_performed",
    "plateau_eligible_for_stopping",
    "plateau_conditions_passed",
    "plateau_qualifying_passed",
    "plateau_consecutive_passing_audits",
    "plateau_relative_mean_improvement",
    "plateau_normalized_absolute_slope_per_epoch",
    "plateau_should_stop",
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finite_float(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise SO1ObservabilityError(f"{field} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise SO1ObservabilityError(f"{field} must be {qualifier}.")
    return result


def _optional_finite(value: object | None, *, field: str) -> float | str:
    if value is None:
        return ""
    result = float(value)
    # A zero previous-window loss can mathematically produce infinity.  Empty
    # is preferable to emitting non-portable CSV spellings such as ``inf``.
    return result if math.isfinite(result) else ""


def plateau_monitor_fields(
    losses: Sequence[float],
    *,
    model_seed: int,
) -> dict[str, object]:
    """Return prespecified plateau fields for any completed epoch.

    An audit occurs only at a 25-epoch boundary with at least 50 observations.
    A stopping decision remains ineligible before epoch 150 and requires two
    consecutive qualifying audits.  These are held-in training diagnostics,
    never validation or checkpoint-selection metrics.
    """

    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise SO1ObservabilityError("Plateau history must be finite and non-empty.")
    completed = int(values.size)
    performed = bool(
        completed >= 50 and completed % CHECKPOINT_INTERVAL_GLOBAL_EPOCHS == 0
    )
    result: dict[str, object] = {
        "plateau_audit_performed": performed,
        "plateau_eligible_for_stopping": completed
        >= PLATEAU_FIRST_ALLOWED_STOP_EPOCH,
        "plateau_conditions_passed": False,
        "plateau_qualifying_passed": False,
        "plateau_consecutive_passing_audits": 0,
        "plateau_relative_mean_improvement": "",
        "plateau_normalized_absolute_slope_per_epoch": "",
        "plateau_should_stop": False,
    }
    if not performed:
        return result
    audit = audit_training_loss_plateau(
        values,
        model_seed=int(model_seed),
        completed_global_epochs=completed,
    )
    result.update(
        {
            "plateau_conditions_passed": bool(audit.conditions_passed),
            "plateau_qualifying_passed": bool(audit.qualifying_passed),
            "plateau_relative_mean_improvement": _optional_finite(
                audit.relative_mean_improvement,
                field="plateau_relative_mean_improvement",
            ),
            "plateau_normalized_absolute_slope_per_epoch": _optional_finite(
                audit.normalized_absolute_slope_per_epoch,
                field="plateau_normalized_absolute_slope_per_epoch",
            ),
        }
    )
    if completed >= PLATEAU_FIRST_ALLOWED_STOP_EPOCH:
        decision = single_seed_plateau_decision(
            values,
            model_seed=int(model_seed),
            completed_global_epochs=completed,
        )
        result["plateau_consecutive_passing_audits"] = int(
            decision.consecutive_passing_audits
        )
        result["plateau_should_stop"] = bool(decision.should_stop)
    return result


def strict_plateau_monitor_fields(
    losses: Sequence[float],
    *,
    model_seed: int,
    audit_interval_global_epochs: int = CHECKPOINT_INTERVAL_GLOBAL_EPOCHS,
    window_global_epochs: int = PLATEAU_WINDOW_GLOBAL_EPOCHS,
    absolute_relative_half_window_change_max: float = (
        STRICT_PLATEAU_ABSOLUTE_RELATIVE_HALF_WINDOW_CHANGE_MAX
    ),
    normalized_absolute_slope_per_epoch_max: float = (
        STRICT_PLATEAU_NORMALIZED_ABSOLUTE_SLOPE_MAX
    ),
    consecutive_passing_audits: int = (
        STRICT_PLATEAU_CONSECUTIVE_PASSING_AUDITS
    ),
) -> dict[str, object]:
    """Return the prespecified strict SO1 training-loss plateau fields.

    Unlike the legacy stopping audit, this diagnostic bounds the *absolute*
    relative change between the two half windows.  A small deterioration can
    therefore no longer pass merely because its signed improvement is
    negative.  ``plateau_should_stop`` and
    ``plateau_eligible_for_stopping`` are always false: these fields reuse the
    established CSV schema.  It becomes eligible to stop training only at or
    after epoch 150 and only after two consecutive passing audit boundaries.
    The loss is held-in training loss, never validation or model-selection
    evidence.
    """

    values = np.asarray(losses, dtype=np.float64)
    interval = int(audit_interval_global_epochs)
    window_size = int(window_global_epochs)
    required_consecutive = int(consecutive_passing_audits)
    relative_limit = float(absolute_relative_half_window_change_max)
    slope_limit = float(normalized_absolute_slope_per_epoch_max)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise SO1ObservabilityError(
            "Strict plateau history must be finite and non-empty."
        )
    if (
        interval <= 0
        or window_size < 2
        or window_size % 2 != 0
        or required_consecutive <= 0
        or not math.isfinite(relative_limit)
        or relative_limit < 0.0
        or not math.isfinite(slope_limit)
        or slope_limit < 0.0
    ):
        raise SO1ObservabilityError("Strict plateau diagnostic settings are invalid.")

    completed = int(values.size)
    performed = completed >= window_size and completed % interval == 0
    result: dict[str, object] = {
        "plateau_audit_performed": bool(performed),
        "plateau_eligible_for_stopping": completed
        >= PLATEAU_FIRST_ALLOWED_STOP_EPOCH,
        "plateau_conditions_passed": False,
        "plateau_qualifying_passed": False,
        "plateau_consecutive_passing_audits": 0,
        "plateau_relative_mean_improvement": "",
        "plateau_normalized_absolute_slope_per_epoch": "",
        "plateau_should_stop": False,
    }
    if not performed:
        return result

    def audit_at(boundary: int) -> tuple[bool, float, float]:
        if boundary < window_size or boundary % interval != 0:
            return False, float("nan"), float("nan")
        window = values[boundary - window_size : boundary]
        half = window_size // 2
        previous_mean = float(window[:half].mean())
        recent_mean = float(window[half:].mean())
        if previous_mean == 0.0:
            relative_change = 0.0 if recent_mean == 0.0 else float("inf")
        else:
            relative_change = (previous_mean - recent_mean) / abs(previous_mean)
        window_mean = float(window.mean())
        x = np.arange(window_size, dtype=np.float64)
        x_centered = x - x.mean()
        slope = float(
            np.dot(x_centered, window - window_mean)
            / np.dot(x_centered, x_centered)
        )
        normalized_slope = (
            abs(slope) / abs(window_mean)
            if window_mean != 0.0
            else (0.0 if slope == 0.0 else float("inf"))
        )
        passed = bool(
            abs(relative_change) <= relative_limit
            and normalized_slope <= slope_limit
        )
        return passed, float(relative_change), float(normalized_slope)

    current_passed, relative_change, normalized_slope = audit_at(completed)
    consecutive = 0
    boundary = completed
    while (
        consecutive < required_consecutive
        and boundary >= PLATEAU_FIRST_ALLOWED_STOP_EPOCH
    ):
        passed, _, _ = audit_at(boundary)
        if not passed:
            break
        consecutive += 1
        boundary -= interval
    result.update(
        {
            "plateau_conditions_passed": current_passed,
            "plateau_qualifying_passed": current_passed,
            "plateau_consecutive_passing_audits": consecutive,
            "plateau_relative_mean_improvement": _optional_finite(
                relative_change,
                field="strict_plateau_relative_half_window_change",
            ),
            "plateau_normalized_absolute_slope_per_epoch": _optional_finite(
                normalized_slope,
                field="strict_plateau_normalized_absolute_slope_per_epoch",
            ),
            "plateau_should_stop": bool(
                completed >= PLATEAU_FIRST_ALLOWED_STOP_EPOCH
                and consecutive >= required_consecutive
            ),
        }
    )
    return result


def build_epoch_metrics_row(
    *,
    run_id: str,
    model_seed: int,
    global_epoch: int,
    equal_core_mean_masked_huber: float,
    per_core_masked_huber: Mapping[str, float],
    gradient_norms_before_clip: Sequence[float],
    learning_rate: float,
    optimizer_updates_this_epoch: int,
    cumulative_optimizer_updates: int,
    epoch_duration_seconds: float,
    prior_epoch_durations: Sequence[float],
    complete_graph_mask_views: int,
    masked_entries_across_views: int,
    peak_vram_gib_all_ranks: float,
    loss_history: Sequence[float],
    strict_plateau_diagnostic: bool = False,
    strict_plateau_audit_interval_global_epochs: int = (
        CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
    ),
    strict_plateau_window_global_epochs: int = PLATEAU_WINDOW_GLOBAL_EPOCHS,
    strict_plateau_absolute_relative_half_window_change_max: float = (
        STRICT_PLATEAU_ABSOLUTE_RELATIVE_HALF_WINDOW_CHANGE_MAX
    ),
    strict_plateau_normalized_absolute_slope_per_epoch_max: float = (
        STRICT_PLATEAU_NORMALIZED_ABSOLUTE_SLOPE_MAX
    ),
    strict_plateau_consecutive_passing_audits: int = (
        STRICT_PLATEAU_CONSECUTIVE_PASSING_AUDITS
    ),
) -> dict[str, object]:
    """Build one complete, finite CSV row for a finished global epoch."""

    epoch = int(global_epoch)
    if epoch <= 0 or len(loss_history) != epoch:
        raise SO1ObservabilityError(
            "global_epoch must equal the cumulative loss-history length."
        )
    if set(per_core_masked_huber) != set(SO1_CORE_ALIASES):
        raise SO1ObservabilityError(
            "Per-core losses require exactly SO1-C01 through SO1-C14."
        )
    gradients = np.asarray(gradient_norms_before_clip, dtype=np.float64)
    if gradients.ndim != 1 or gradients.size != 7 or not np.isfinite(
        gradients
    ).all() or bool((gradients < 0).any()):
        raise SO1ObservabilityError(
            "Exactly seven finite non-negative update gradient norms are required."
        )
    duration = _finite_float(
        epoch_duration_seconds,
        field="epoch_duration_seconds",
        nonnegative=True,
    )
    durations = [
        _finite_float(value, field="prior_epoch_duration", nonnegative=True)
        for value in prior_epoch_durations
    ] + [duration]
    rolling_duration = float(np.mean(durations[-5:]))
    complete_views = int(complete_graph_mask_views)
    masked_entries = int(masked_entries_across_views)
    updates = int(optimizer_updates_this_epoch)
    cumulative_updates = int(cumulative_optimizer_updates)
    if complete_views != 140:
        raise SO1ObservabilityError(
            "A completed SO1 epoch must contain exactly 140 graph-mask views."
        )
    if masked_entries < 0:
        raise SO1ObservabilityError("Masked entry count cannot be negative.")
    if updates != 7 or cumulative_updates != epoch * 7:
        raise SO1ObservabilityError(
            "A completed SO1 epoch must add seven cumulative optimizer updates."
        )
    next_audit = (
        ((epoch // CHECKPOINT_INTERVAL_GLOBAL_EPOCHS) + 1)
        * CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
    )
    plateau_fields = (
        strict_plateau_monitor_fields(
            loss_history,
            model_seed=int(model_seed),
            audit_interval_global_epochs=(
                strict_plateau_audit_interval_global_epochs
            ),
            window_global_epochs=strict_plateau_window_global_epochs,
            absolute_relative_half_window_change_max=(
                strict_plateau_absolute_relative_half_window_change_max
            ),
            normalized_absolute_slope_per_epoch_max=(
                strict_plateau_normalized_absolute_slope_per_epoch_max
            ),
            consecutive_passing_audits=(
                strict_plateau_consecutive_passing_audits
            ),
        )
        if strict_plateau_diagnostic
        else plateau_monitor_fields(loss_history, model_seed=int(model_seed))
    )
    row: dict[str, object] = {
        "schema": EPOCH_METRICS_SCHEMA,
        "run_id": str(run_id),
        "model_seed": int(model_seed),
        "global_epoch": epoch,
        "equal_core_mean_masked_huber": _finite_float(
            equal_core_mean_masked_huber,
            field="equal_core_mean_masked_huber",
        ),
        "gradient_norm_mean_before_clip": float(gradients.mean()),
        "gradient_norm_max_before_clip": float(gradients.max()),
        "learning_rate": _finite_float(learning_rate, field="learning_rate"),
        "optimizer_updates_this_epoch": updates,
        "cumulative_optimizer_updates": cumulative_updates,
        "epoch_duration_seconds": duration,
        "rolling_5_epoch_duration_seconds": rolling_duration,
        "elapsed_training_seconds": float(sum(durations)),
        "eta_to_next_plateau_audit_seconds": float(
            (next_audit - epoch) * rolling_duration
        ),
        "eta_to_epoch_200_seconds": float(max(0, 200 - epoch) * rolling_duration),
        "complete_graph_mask_views": complete_views,
        "masked_entries_across_views": masked_entries,
        "complete_graph_views_per_second": (
            0.0 if duration == 0.0 else float(complete_views / duration)
        ),
        "masked_entries_per_second": (
            0.0 if duration == 0.0 else float(masked_entries / duration)
        ),
        "peak_vram_gib_all_ranks": _finite_float(
            peak_vram_gib_all_ranks,
            field="peak_vram_gib_all_ranks",
            nonnegative=True,
        ),
        **plateau_fields,
    }
    for alias in SO1_CORE_ALIASES:
        row[_loss_column(alias)] = _finite_float(
            per_core_masked_huber[alias], field=f"loss[{alias}]"
        )
    if tuple(row) != EPOCH_METRICS_COLUMNS:
        # Dict insertion differs because core columns were populated last.
        row = {column: row[column] for column in EPOCH_METRICS_COLUMNS}
    return row


def _normalized_csv_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


class DurableEpochMetricsCSV:
    """Atomically publish one fsynced row per epoch with resume reconciliation."""

    def __init__(self, run_root: str | Path, *, rank: int = 0) -> None:
        if int(rank) != 0:
            raise SO1ObservabilityError("Only distributed rank zero may write metrics.")
        self.run_root = Path(run_root).resolve(strict=True)
        self.path = self.run_root / "results" / "epoch_metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise SO1ObservabilityError("Epoch metrics path must be a regular file.")
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EPOCH_METRICS_COLUMNS:
                raise SO1ObservabilityError("Epoch metrics CSV schema mismatch.")
            rows = list(reader)
        observed: list[int] = []
        for row in rows:
            try:
                observed.append(int(row["global_epoch"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise SO1ObservabilityError(
                    "Epoch metrics CSV contains an invalid epoch."
                ) from exc
        expected = list(range(1, len(rows) + 1))
        if observed != expected:
            raise SO1ObservabilityError(
                "Epoch metrics CSV must contain unique contiguous epochs from one."
            )
        return rows

    @property
    def completed_epochs(self) -> int:
        return len(self._read_rows())

    def read_rows(self) -> tuple[dict[str, str], ...]:
        """Return a validated snapshot for monitoring and final summaries."""

        return tuple(self._read_rows())

    @property
    def prior_durations(self) -> tuple[float, ...]:
        rows = self._read_rows()
        try:
            return tuple(float(row["epoch_duration_seconds"]) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise SO1ObservabilityError(
                "Epoch metrics CSV contains an invalid duration."
            ) from exc

    def reconcile(self, *, checkpoint_epoch: int) -> int:
        """Trim rows newer than a durable checkpoint after an interrupted epoch.

        The append-only scientific event log is not rewritten.  CSV is a live
        operational projection and may be atomically truncated only to the
        last checksum-valid checkpoint before deterministic replay.
        """

        durable_epoch = int(checkpoint_epoch)
        if durable_epoch < 0:
            raise SO1ObservabilityError("Checkpoint epoch cannot be negative.")
        rows = self._read_rows()
        if len(rows) < durable_epoch:
            raise SO1ObservabilityError(
                "Epoch metrics end before the resume checkpoint; copy the source "
                "CSV or reconstruct its rows before resuming."
            )
        if len(rows) == durable_epoch:
            return durable_epoch
        self._atomic_rewrite(rows[:durable_epoch])
        return durable_epoch

    def _atomic_rewrite(self, rows: Sequence[Mapping[str, object]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".epoch_metrics.csv.", suffix=".writing", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=EPOCH_METRICS_COLUMNS)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            column: _normalized_csv_value(row[column])
                            for column in EPOCH_METRICS_COLUMNS
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def append(self, row: Mapping[str, object]) -> bool:
        """Atomically add one row; return false for an exact idempotent replay."""

        if set(row) != set(EPOCH_METRICS_COLUMNS):
            raise SO1ObservabilityError("Epoch metrics row has the wrong columns.")
        rows = self._read_rows()
        epoch = int(row["global_epoch"])
        expected = len(rows) + 1
        normalized = {
            column: _normalized_csv_value(row[column])
            for column in EPOCH_METRICS_COLUMNS
        }
        if epoch == len(rows) and rows:
            if rows[-1] == normalized:
                return False
            raise SO1ObservabilityError(
                "Refusing a divergent duplicate epoch metrics row."
            )
        if epoch != expected:
            raise SO1ObservabilityError(
                f"Expected metrics epoch {expected}; received {epoch}."
            )
        # Publish the header and every complete row as one atomic snapshot.  A
        # process or host failure before ``os.replace`` leaves the prior CSV
        # intact (or no CSV for the first epoch), so readers can never observe
        # a partially written header or trailing row.
        self._atomic_rewrite((*rows, normalized))
        return True


def epoch_event_already_recorded(run_root: str | Path, *, epoch: int) -> bool:
    """Detect a prior epoch event without mutating the append-only JSONL log."""

    path = Path(run_root) / "metrics" / "events.jsonl"
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SO1ObservabilityError(
                    "Metric event log contains invalid JSON."
                ) from exc
            if (
                isinstance(value, Mapping)
                and value.get("name")
                == "fit/training/equal_core_mean_masked_huber"
                and int(value.get("step", -1)) == int(epoch)
            ):
                return True
    return False


def torch_checkpoint_bytes(payload: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    return stream.getvalue()


@dataclass(frozen=True)
class CheckpointWriteReceipt:
    path: Path
    completed_global_epochs: int
    sha256: str
    size_bytes: int
    role: str


class AtomicLatestCheckpointStore:
    """Atomically replace one resumable checkpoint and finalize it as ``last``."""

    def __init__(
        self,
        run_root: str | Path,
        *,
        rank: int = 0,
        completed_global_epochs: int = 0,
    ) -> None:
        if int(rank) != 0:
            raise SO1ObservabilityError(
                "Only distributed rank zero may write checkpoints."
            )
        self.run_root = Path(run_root).resolve(strict=True)
        self.directory = self.run_root / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.ckpt"
        self.last_path = self.directory / "last.ckpt"
        self.completed_global_epochs = int(completed_global_epochs)
        if self.completed_global_epochs < 0:
            raise SO1ObservabilityError("Completed checkpoint epoch cannot be negative.")
        if self.latest_path.exists() and self.last_path.exists():
            raise SO1ObservabilityError(
                "Checkpoint directory cannot contain both latest.ckpt and last.ckpt."
            )
        unexpected = sorted(
            path.name
            for path in self.directory.glob("*.ckpt")
            if path not in {self.latest_path, self.last_path}
        )
        if unexpected:
            raise SO1ObservabilityError(
                "Latest-only checkpoint directory contains: " + ", ".join(unexpected)
            )

    @staticmethod
    def _validate_loaded(
        path: Path,
        *,
        expected_epoch: int | None = None,
    ) -> dict[str, Any]:
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise SO1ObservabilityError(f"Checkpoint is not reloadable: {path}.") from exc
        if not isinstance(value, Mapping):
            raise SO1ObservabilityError("Checkpoint payload must be a mapping.")
        payload = dict(value)
        if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
            raise SO1ObservabilityError("Checkpoint schema mismatch.")
        try:
            epoch = int(payload["completed_global_epochs"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SO1ObservabilityError(
                "Checkpoint lacks a valid completed epoch."
            ) from exc
        if epoch <= 0 or (expected_epoch is not None and epoch != expected_epoch):
            raise SO1ObservabilityError("Checkpoint completed epoch mismatch.")
        return payload

    def _write_bytes(
        self,
        content: bytes,
        *,
        epoch: int,
        allow_same_epoch: bool = False,
    ) -> CheckpointWriteReceipt:
        if self.last_path.exists():
            raise SO1ObservabilityError("A finalized last checkpoint is immutable.")
        epoch_is_allowed = (
            epoch == self.completed_global_epochs
            if allow_same_epoch
            else epoch > self.completed_global_epochs
        )
        if not epoch_is_allowed:
            raise SO1ObservabilityError(
                "Rolling checkpoint epoch does not satisfy the replacement policy."
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".latest.ckpt.", suffix=".writing", dir=self.directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            self._validate_loaded(temporary, expected_epoch=epoch)
            os.replace(temporary, self.latest_path)
            _fsync_directory(self.directory)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.completed_global_epochs = epoch
        return CheckpointWriteReceipt(
            path=self.latest_path,
            completed_global_epochs=epoch,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            role="latest",
        )

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        completed_global_epochs: int,
    ) -> CheckpointWriteReceipt:
        epoch = int(completed_global_epochs)
        if int(payload.get("completed_global_epochs", -1)) != epoch:
            raise SO1ObservabilityError(
                "Checkpoint payload and write epoch do not agree."
            )
        return self._write_bytes(torch_checkpoint_bytes(payload), epoch=epoch)

    def load_latest(self) -> dict[str, Any]:
        if not self.latest_path.is_file() or self.latest_path.is_symlink():
            raise SO1ObservabilityError("No regular latest checkpoint is available.")
        return self._validate_loaded(
            self.latest_path,
            expected_epoch=self.completed_global_epochs,
        )

    def finalize(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        completed_global_epochs: int | None = None,
    ) -> CheckpointWriteReceipt:
        """Optionally replace latest with a final payload, then rename it atomically."""

        if self.last_path.exists():
            raise SO1ObservabilityError("Final checkpoint already exists.")
        if payload is not None:
            if completed_global_epochs is None:
                raise SO1ObservabilityError("Final payload requires its epoch.")
            # A final payload usually describes the same epoch as latest.  It
            # may add the plateau decision without advancing training.
            epoch = int(completed_global_epochs)
            if epoch != self.completed_global_epochs:
                raise SO1ObservabilityError(
                    "Final checkpoint must describe the current latest epoch."
                )
            if int(payload.get("completed_global_epochs", -1)) != epoch:
                raise SO1ObservabilityError(
                    "Final checkpoint payload and epoch do not agree."
                )
            self._write_bytes(
                torch_checkpoint_bytes(payload),
                epoch=epoch,
                allow_same_epoch=True,
            )
        if not self.latest_path.is_file() or self.latest_path.is_symlink():
            raise SO1ObservabilityError("Cannot finalize without latest.ckpt.")
        self._validate_loaded(
            self.latest_path,
            expected_epoch=self.completed_global_epochs,
        )
        os.replace(self.latest_path, self.last_path)
        _fsync_directory(self.directory)
        content_sha256 = hashlib.sha256()
        size = 0
        with self.last_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                content_sha256.update(chunk)
        return CheckpointWriteReceipt(
            path=self.last_path,
            completed_global_epochs=self.completed_global_epochs,
            sha256=content_sha256.hexdigest(),
            size_bytes=size,
            role="last",
        )


def checkpoint_payload_epoch(payload: Mapping[str, Any]) -> int:
    """Small public extractor useful for runner resume reconciliation."""

    try:
        return int(payload["completed_global_epochs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO1ObservabilityError("Checkpoint lacks completed_global_epochs.") from exc


def require_latest_only_checkpoint_layout(run_root: str | Path, *, final: bool) -> Path:
    """Verify the exact active or final checkpoint layout."""

    directory = Path(run_root) / "checkpoints"
    files = sorted(path for path in directory.glob("*.ckpt") if path.is_file())
    expected_name = "last.ckpt" if final else "latest.ckpt"
    if len(files) != 1 or files[0].name != expected_name:
        raise SO1ObservabilityError(
            f"Expected only checkpoints/{expected_name}; found "
            + ", ".join(path.name for path in files)
        )
    return files[0]

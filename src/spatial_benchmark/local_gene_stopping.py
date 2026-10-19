"""Validation-NLL stopping with separate checkpoint and patience criteria.

This pure-Python policy never reads training metrics or controls learning rates.
Call it exactly once after each complete, finite validation epoch.  The trainer
owns its independent ReduceLROnPlateau scheduler and checkpoint writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from numbers import Integral, Real
from typing import Any, Mapping


SCHEMA = "local_gene_early_stopping_v1"


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


@dataclass(frozen=True)
class LocalGeneStoppingConfig:
    patience: int = 30
    min_delta: float = 1e-4
    min_epochs: int = 50
    max_epochs: int = 500


@dataclass(frozen=True)
class LocalGeneStoppingDecision:
    epoch: int
    count_nll: float
    improved_best: bool
    improved_anchor: bool
    best_epoch: int
    best_count_nll: float
    anchor_epoch: int
    anchor_count_nll: float
    epochs_without_improvement: int
    should_stop: bool
    stop_reason: str | None
    max_epochs_reached: bool


class LocalGeneEarlyStopping:
    """Track strict checkpoint minima and cumulative material improvement.

    ``improved_best`` is true whenever NLL is strictly below every previous
    epoch; ties retain the earlier checkpoint.  Patience resets only when NLL
    reaches ``anchor_count_nll - min_delta``.  Several smaller strict minima
    can therefore accumulate into one significant improvement.

    Patience accrues from the first observation, including epochs below the
    minimum.  ``min_epochs`` only gates stopping: with the default policy,
    constant validation NLL stops at epoch 50.  A significant improvement at
    epoch 50 postpones a patience stop until epoch 80.  The maximum is a safety
    cap, explicitly distinguished from patience exhaustion.  When both become
    true on the same epoch, patience takes precedence and the cap flag is true.
    """

    def __init__(
        self,
        *,
        patience: int = 30,
        min_delta: float = 1e-4,
        min_epochs: int = 50,
        max_epochs: int = 500,
    ) -> None:
        patience = _integer(patience, "patience", 1)
        minimum = _integer(min_epochs, "min_epochs", 1)
        maximum = _integer(max_epochs, "max_epochs", 1)
        delta = _finite(min_delta, "min_delta")
        if delta <= 0:
            raise ValueError("min_delta must be strictly positive")
        if minimum > maximum:
            raise ValueError("min_epochs cannot exceed max_epochs")
        self.config = LocalGeneStoppingConfig(patience, delta, minimum, maximum)
        self._decision: LocalGeneStoppingDecision | None = None

    @property
    def completed_epoch(self) -> int:
        return 0 if self._decision is None else self._decision.epoch

    @property
    def should_stop(self) -> bool:
        return self._decision is not None and self._decision.should_stop

    @property
    def stop_reason(self) -> str | None:
        return None if self._decision is None else self._decision.stop_reason

    def _stop_reason(self, epoch: int, bad_epochs: int) -> str | None:
        if epoch >= self.config.min_epochs and bad_epochs >= self.config.patience:
            return "patience"
        if epoch >= self.config.max_epochs:
            return "maximum_epochs"
        return None

    def observe(self, *, epoch: int, count_nll: float) -> LocalGeneStoppingDecision:
        """Advance one complete validation; invalid input leaves state unchanged."""

        if self.should_stop:
            raise RuntimeError("cannot observe another epoch after stopping")
        epoch = _integer(epoch, "epoch", 1)
        if epoch != self.completed_epoch + 1:
            raise ValueError("validation epochs must be consecutive and start at one")
        value = _finite(count_nll, "count_nll")
        if value < 0:
            raise ValueError("full-constant count_nll cannot be negative")
        old = self._decision
        improved_best = old is None or value < old.best_count_nll
        # Compare to a precomputed threshold rather than subtracting two close
        # NLLs, so equality at the representable boundary counts as improvement.
        improved_anchor = old is None or (
            value < old.anchor_count_nll
            and value <= old.anchor_count_nll - self.config.min_delta
        )
        best_value = value if improved_best else old.best_count_nll
        best_epoch = epoch if improved_best else old.best_epoch
        anchor_value = value if improved_anchor else old.anchor_count_nll
        anchor_epoch = epoch if improved_anchor else old.anchor_epoch
        bad_epochs = epoch - anchor_epoch
        reason = self._stop_reason(epoch, bad_epochs)
        decision = LocalGeneStoppingDecision(
            epoch=epoch,
            count_nll=value,
            improved_best=improved_best,
            improved_anchor=improved_anchor,
            best_epoch=best_epoch,
            best_count_nll=best_value,
            anchor_epoch=anchor_epoch,
            anchor_count_nll=anchor_value,
            epochs_without_improvement=bad_epochs,
            should_stop=reason is not None,
            stop_reason=reason,
            max_epochs_reached=epoch >= self.config.max_epochs,
        )
        self._decision = decision
        return decision

    def state_dict(self) -> dict[str, Any]:
        """Return independent JSON-safe policy/state for a checkpoint."""

        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "state": None if self._decision is None else asdict(self._decision),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> LocalGeneEarlyStopping:
        """Restore with strict schema, finite-value, and transition validation."""

        if not isinstance(payload, Mapping) or set(payload) != {"schema", "config", "state"}:
            raise ValueError("invalid stopping state schema fields")
        if payload["schema"] != SCHEMA:
            raise ValueError("unknown stopping state schema")
        config = payload["config"]
        if not isinstance(config, Mapping) or set(config) != {field.name for field in fields(LocalGeneStoppingConfig)}:
            raise ValueError("invalid stopping configuration fields")
        result = cls(**dict(config))
        state = payload["state"]
        if state is None:
            return result
        if not isinstance(state, Mapping) or set(state) != {field.name for field in fields(LocalGeneStoppingDecision)}:
            raise ValueError("invalid stopping decision fields")
        values = dict(state)
        for name in ("epoch", "best_epoch", "anchor_epoch"):
            values[name] = _integer(values[name], name, 1)
        values["epochs_without_improvement"] = _integer(
            values["epochs_without_improvement"], "epochs_without_improvement", 0
        )
        for name in ("count_nll", "best_count_nll", "anchor_count_nll"):
            values[name] = _finite(values[name], name)
            if values[name] < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("improved_best", "improved_anchor", "should_stop", "max_epochs_reached"):
            if not isinstance(values[name], bool):
                raise ValueError(f"{name} must be boolean")
        epoch, best_epoch, anchor_epoch = values["epoch"], values["best_epoch"], values["anchor_epoch"]
        if not 1 <= anchor_epoch <= best_epoch <= epoch <= result.config.max_epochs:
            raise ValueError("stopping epoch ordering is inconsistent")
        best, anchor, current = values["best_count_nll"], values["anchor_count_nll"], values["count_nll"]
        if best > anchor or best > current or (best < anchor and best <= anchor - result.config.min_delta):
            raise ValueError("stopping minimum/anchor metrics are inconsistent")
        if (best_epoch == anchor_epoch) != (best == anchor):
            raise ValueError("stopping minimum/anchor epochs disagree with their metrics")
        if best_epoch == epoch and current != best:
            raise ValueError("current strict minimum does not match count_nll")
        if anchor_epoch == epoch and current != anchor:
            raise ValueError("current anchor does not match count_nll")
        bad_epochs = epoch - anchor_epoch
        if values["epochs_without_improvement"] != bad_epochs:
            raise ValueError("stopping bad-epoch counter disagrees with anchor")
        if values["improved_best"] != (best_epoch == epoch) or values["improved_anchor"] != (anchor_epoch == epoch):
            raise ValueError("stopping improvement flags disagree with epochs")
        reason = result._stop_reason(epoch, bad_epochs)
        if values["stop_reason"] != reason or values["should_stop"] != (reason is not None):
            raise ValueError("stopping outcome disagrees with policy")
        if values["max_epochs_reached"] != (epoch == result.config.max_epochs):
            raise ValueError("stopping safety-cap flag disagrees with epoch")
        result._decision = LocalGeneStoppingDecision(**values)
        return result

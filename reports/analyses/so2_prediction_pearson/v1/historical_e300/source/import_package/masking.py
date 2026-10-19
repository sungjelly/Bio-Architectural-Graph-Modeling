"""Deterministic expression masking for the spatial benchmark.

The functions in this module deliberately know nothing about model seeds.  A
mask seed is a separate experimental input so that architectures and model
initialisations can be compared with exactly paired training and evaluation
masks.

Only the expression matrix is maskable.  Metadata can be passed to
``apply_expression_mask`` for convenience, but is copied without alteration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MASK_BUNDLE_FORMAT_VERSION = 1
_MASK_MODES = {
    "p": "partial",
    "partial": "partial",
    "partial_gene": "partial",
    "partial-gene": "partial",
    "partial_genes": "partial",
    "n": "node",
    "node": "node",
    "whole_node": "node",
    "whole-node": "node",
    "b": "block",
    "block": "block",
    "spatial_block": "block",
    "spatial-block": "block",
}


def _canonical_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    try:
        return _MASK_MODES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_MASK_MODES.values())))
        raise ValueError(
            f"Unknown mask mode {mode!r}; expected one of {allowed}"
        ) from exc


def _validate_rate(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return value


@dataclass(frozen=True)
class MaskSpec:
    """Configuration for one expression masking estimand.

    ``block_width_um`` is a physical diameter/side width.  With ``disk`` it is
    the diameter; with ``square`` it is the side length.  When it is ``None``,
    the block is the physical disk around a sampled anchor that contains at
    least ``block_node_rate`` of eligible cells.
    """

    mode: str
    partial_gene_rate: float = 0.20
    node_rate: float = 0.10
    block_node_rate: float = 0.10
    block_width_um: float | None = None
    block_shape: str = "disk"
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _canonical_mode(self.mode))
        object.__setattr__(
            self,
            "partial_gene_rate",
            _validate_rate("partial_gene_rate", self.partial_gene_rate),
        )
        object.__setattr__(
            self,
            "node_rate",
            _validate_rate("node_rate", self.node_rate),
        )
        object.__setattr__(
            self,
            "block_node_rate",
            _validate_rate("block_node_rate", self.block_node_rate),
        )
        shape = str(self.block_shape).strip().lower()
        if shape not in {"disk", "square"}:
            raise ValueError("block_shape must be 'disk' or 'square'")
        object.__setattr__(self, "block_shape", shape)
        if self.block_width_um is not None:
            width = float(self.block_width_um)
            if not math.isfinite(width) or width <= 0:
                raise ValueError("block_width_um must be finite and positive")
            object.__setattr__(self, "block_width_um", width)
        if self.label is not None:
            label = str(self.label).strip()
            if not label:
                raise ValueError("label must be non-empty when supplied")
            object.__setattr__(self, "label", label)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)

    @property
    def spec_id(self) -> str:
        """Stable ID that changes whenever a masking parameter changes."""

        digest = hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()[:12]
        prefix = (
            re.sub(r"[^a-z0-9]+", "-", self.label.lower())
            if self.label
            else self.mode
        )
        return f"{prefix.strip('-')}-{digest}"


@dataclass(frozen=True)
class MaskedInputs:
    """Expression, explicit gene-mask channel, and untouched metadata."""

    expression: Any
    gene_mask: Any
    metadata: Any = None


@dataclass(frozen=True)
class MaskBatch:
    """A generated boolean expression mask and its provenance."""

    mask: np.ndarray
    spec: MaskSpec
    seed: int
    eligible_nodes: np.ndarray
    selected_nodes: np.ndarray
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        eligible = np.asarray(self.eligible_nodes)
        selected = np.asarray(self.selected_nodes)
        if mask.ndim != 2 or mask.dtype != np.bool_:
            raise TypeError("mask must be a boolean [n_nodes, n_genes] array")
        if eligible.shape != (mask.shape[0],) or eligible.dtype != np.bool_:
            raise TypeError("eligible_nodes must be a boolean [n_nodes] array")
        if selected.shape != (mask.shape[0],) or selected.dtype != np.bool_:
            raise TypeError("selected_nodes must be a boolean [n_nodes] array")
        if np.any(selected & ~eligible):
            raise ValueError("selected_nodes contains ineligible cells")
        if np.any(mask[~eligible]):
            raise ValueError("ineligible cells may not be masked")
        if not np.array_equal(mask.any(axis=1), selected):
            raise ValueError("selected_nodes must equal mask.any(axis=1)")
        for name, value in (
            ("mask", mask),
            ("eligible_nodes", eligible),
            ("selected_nodes", selected),
        ):
            value = np.ascontiguousarray(value.copy())
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "details", copy.deepcopy(dict(self.details)))

    @property
    def n_nodes(self) -> int:
        return int(self.mask.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.mask.shape[1])

    @property
    def n_eligible_nodes(self) -> int:
        return int(self.eligible_nodes.sum())

    @property
    def n_selected_nodes(self) -> int:
        return int(self.selected_nodes.sum())

    @property
    def n_masked_entries(self) -> int:
        return int(self.mask.sum())

    @property
    def achieved_node_rate(self) -> float:
        if self.n_eligible_nodes == 0:
            return 0.0
        return self.n_selected_nodes / self.n_eligible_nodes

    @property
    def achieved_entry_rate(self) -> float:
        denominator = self.n_eligible_nodes * self.n_genes
        return self.n_masked_entries / denominator if denominator else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.spec.mode,
            "seed": self.seed,
            "shape": [self.n_nodes, self.n_genes],
            "n_eligible_nodes": self.n_eligible_nodes,
            "n_selected_nodes": self.n_selected_nodes,
            "n_masked_entries": self.n_masked_entries,
            "achieved_node_rate": self.achieved_node_rate,
            "achieved_entry_rate": self.achieved_entry_rate,
            "details": copy.deepcopy(dict(self.details)),
        }

    def apply(
        self,
        expression: Any,
        metadata: Any = None,
        *,
        fill_value: float = 0.0,
    ) -> MaskedInputs:
        """Apply this expression mask while leaving metadata untouched."""

        return apply_expression_mask(
            expression,
            self.mask,
            metadata=metadata,
            fill_value=fill_value,
        )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def derive_mask_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable 63-bit mask seed without Python's salted ``hash``.

    Model initialisation seeds are intentionally absent from this API.
    """

    payload = {"base_seed": int(base_seed), "parts": list(parts)}
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def paired_epoch_seed(mask_seed: int, epoch: int, batch_index: int = 0) -> int:
    """Return the paired mask seed for an epoch/batch across all models."""

    if int(epoch) < 0 or int(batch_index) < 0:
        raise ValueError("epoch and batch_index must be non-negative")
    return derive_mask_seed(int(mask_seed), "train", int(epoch), int(batch_index))


def curriculum_mode(
    epoch: int,
    seed: int,
    curriculum: str = "P+N+B",
    warmup_epochs: int = 10,
    *,
    batch_index: int = 0,
) -> str:
    """Select a deterministic paired training-mask mode.

    The first ``warmup_epochs`` always use partial-gene masking.  Thereafter:

    - ``P-only``: 100% partial;
    - ``P+N``: 70% partial, 30% whole-node;
    - ``P+N+B``: 60% partial, 30% whole-node, 10% spatial block.

    ``seed`` is the masking-experiment seed, not a model seed.
    """

    epoch = int(epoch)
    batch_index = int(batch_index)
    warmup_epochs = int(warmup_epochs)
    if epoch < 0 or batch_index < 0 or warmup_epochs < 0:
        raise ValueError("epoch, batch_index, and warmup_epochs must be non-negative")
    if epoch < warmup_epochs:
        return "partial"

    key = re.sub(r"[\s_]+", "", str(curriculum)).upper()
    schedules: dict[str, tuple[tuple[str, float], ...]] = {
        "P": (("partial", 1.0),),
        "P-ONLY": (("partial", 1.0),),
        "PONLY": (("partial", 1.0),),
        "P+N": (("partial", 0.70), ("node", 0.30)),
        "P+N+B": (("partial", 0.60), ("node", 0.30), ("block", 0.10)),
    }
    if key not in schedules:
        raise ValueError("curriculum must be one of P-only, P+N, or P+N+B")
    draw_seed = derive_mask_seed(seed, "curriculum", epoch, batch_index)
    draw = float(np.random.default_rng(draw_seed).random())
    cumulative = 0.0
    for mode, probability in schedules[key]:
        cumulative += probability
        if draw < cumulative:
            return mode
    return schedules[key][-1][0]  # guard against floating-point endpoint effects


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalise_coordinates(coordinates_um: Any) -> np.ndarray:
    coordinates = _as_numpy(coordinates_um)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("coordinates_um must have shape [n_nodes, >=2]")
    coordinates = np.asarray(coordinates[:, :2], dtype=np.float64)
    return np.ascontiguousarray(coordinates)


def _normalise_eligible(eligible_nodes: Any, n_nodes: int) -> np.ndarray:
    if eligible_nodes is None:
        return np.ones(n_nodes, dtype=bool)
    eligible = _as_numpy(eligible_nodes)
    if eligible.dtype == np.bool_:
        if eligible.shape != (n_nodes,):
            raise ValueError("boolean eligible_nodes must have shape [n_nodes]")
        return np.asarray(eligible, dtype=bool).copy()
    if eligible.ndim != 1 or not np.issubdtype(eligible.dtype, np.integer):
        raise TypeError("eligible_nodes must be a boolean mask or integer indices")
    result = np.zeros(n_nodes, dtype=bool)
    indices = np.asarray(eligible, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= n_nodes):
        raise IndexError("eligible node index is out of bounds")
    result[indices] = True
    return result


def _count_from_rate(n: int, rate: float) -> int:
    if n <= 0 or rate <= 0:
        return 0
    return min(n, max(1, int(math.floor(n * rate + 0.5))))


def _resolve_spec(
    mode: str | MaskSpec,
    *,
    partial_gene_rate: float | None,
    node_rate: float | None,
    block_node_rate: float | None,
    block_width_um: float | None,
    block_shape: str | None,
) -> MaskSpec:
    if isinstance(mode, MaskSpec):
        values = mode.to_dict()
        if partial_gene_rate is not None:
            values["partial_gene_rate"] = partial_gene_rate
        if node_rate is not None:
            values["node_rate"] = node_rate
        if block_node_rate is not None:
            values["block_node_rate"] = block_node_rate
        if block_width_um is not None:
            values["block_width_um"] = block_width_um
        if block_shape is not None:
            values["block_shape"] = block_shape
        return MaskSpec(**values)
    return MaskSpec(
        mode=mode,
        partial_gene_rate=0.20 if partial_gene_rate is None else partial_gene_rate,
        node_rate=0.10 if node_rate is None else node_rate,
        block_node_rate=0.10 if block_node_rate is None else block_node_rate,
        block_width_um=block_width_um,
        block_shape="disk" if block_shape is None else block_shape,
    )


def generate_mask(
    mode: str | MaskSpec,
    n_genes: int,
    coordinates_um: Any,
    eligible_nodes: Any = None,
    seed: int = 0,
    *,
    partial_gene_rate: float | None = None,
    node_rate: float | None = None,
    block_node_rate: float | None = None,
    block_width_um: float | None = None,
    block_shape: str | None = None,
) -> MaskBatch:
    """Generate a deterministic boolean ``[n_nodes, n_genes]`` mask.

    A partial mask hides the same exact number of randomly selected genes in
    every eligible cell.  A node mask samples eligible cells and hides every
    gene in them.  A block mask samples a physical anchor and hides every gene
    in cells inside one contiguous disk or square.
    """

    n_genes = int(n_genes)
    if n_genes <= 0:
        raise ValueError("n_genes must be positive")
    seed = int(seed)
    spec = _resolve_spec(
        mode,
        partial_gene_rate=partial_gene_rate,
        node_rate=node_rate,
        block_node_rate=block_node_rate,
        block_width_um=block_width_um,
        block_shape=block_shape,
    )
    coordinates = _normalise_coordinates(coordinates_um)
    n_nodes = coordinates.shape[0]
    eligible = _normalise_eligible(eligible_nodes, n_nodes)
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size and not np.all(np.isfinite(coordinates[eligible])):
        raise ValueError("eligible coordinates must be finite")

    rng = np.random.default_rng(seed)
    mask = np.zeros((n_nodes, n_genes), dtype=bool)
    details: dict[str, Any] = {}

    if spec.mode == "partial":
        n_hidden = _count_from_rate(n_genes, spec.partial_gene_rate)
        if n_hidden == n_genes:
            mask[eligible] = True
        elif n_hidden:
            # Chunking avoids allocating an n_cells x n_genes float64 matrix for
            # the full core while retaining an exact per-cell mask rate.
            chunk_size = max(1, min(2048, 16_000_000 // max(1, n_genes)))
            for start in range(0, eligible_indices.size, chunk_size):
                rows = eligible_indices[start : start + chunk_size]
                scores = rng.random((rows.size, n_genes), dtype=np.float32)
                columns = np.argpartition(
                    scores,
                    kth=n_hidden - 1,
                    axis=1,
                )[:, :n_hidden]
                mask[rows[:, None], columns] = True
        details = {
            "requested_gene_rate": spec.partial_gene_rate,
            "genes_per_selected_node": n_hidden,
            "achieved_gene_rate_per_selected_node": n_hidden / n_genes,
        }

    elif spec.mode == "node":
        n_hidden_nodes = _count_from_rate(eligible_indices.size, spec.node_rate)
        if n_hidden_nodes:
            chosen = np.asarray(
                rng.choice(eligible_indices, size=n_hidden_nodes, replace=False),
                dtype=np.int64,
            )
            mask[chosen] = True
        details = {
            "requested_node_rate": spec.node_rate,
            "requested_nodes": n_hidden_nodes,
        }

    else:
        if eligible_indices.size:
            anchor = int(rng.choice(eligible_indices))
            center = coordinates[anchor]
            offsets = coordinates[eligible_indices] - center
            distances = np.sqrt(np.einsum("ij,ij->i", offsets, offsets))
            if spec.block_width_um is None:
                requested = _count_from_rate(
                    eligible_indices.size,
                    spec.block_node_rate,
                )
                if requested:
                    kth_distance = float(
                        np.partition(distances, requested - 1)[requested - 1]
                    )
                    tolerance = np.finfo(np.float64).eps * max(1.0, kth_distance) * 8
                    inside = distances <= kth_distance + tolerance
                    chosen = eligible_indices[inside]
                else:
                    kth_distance = 0.0
                    chosen = np.empty(0, dtype=np.int64)
                region_width = 2.0 * kth_distance
                region_rule = "nearest_disk"
            else:
                half_width = spec.block_width_um / 2.0
                if spec.block_shape == "disk":
                    inside = distances <= half_width
                else:
                    inside = np.max(np.abs(offsets), axis=1) <= half_width
                chosen = eligible_indices[inside]
                region_width = spec.block_width_um
                region_rule = spec.block_shape
            mask[chosen] = True
            details = {
                "requested_node_rate": spec.block_node_rate,
                "block_width_um": region_width,
                "block_shape": region_rule,
                "anchor_node": anchor,
                "center_um": [float(center[0]), float(center[1])],
            }
        else:
            details = {
                "requested_node_rate": spec.block_node_rate,
                "block_width_um": spec.block_width_um,
                "block_shape": spec.block_shape,
                "anchor_node": None,
                "center_um": None,
            }

    selected = mask.any(axis=1)
    batch = MaskBatch(
        mask=mask,
        spec=spec,
        seed=seed,
        eligible_nodes=eligible,
        selected_nodes=selected,
        details=details,
    )
    # The achieved rate is material for fixed-width blocks and belongs in every
    # manifest rather than being reconstructed from a requested width.
    batch.details["achieved_node_rate"] = batch.achieved_node_rate
    batch.details["achieved_entry_rate"] = batch.achieved_entry_rate
    return batch


def apply_expression_mask(
    expression: Any,
    mask: Any,
    metadata: Any = None,
    *,
    fill_value: float = 0.0,
) -> MaskedInputs:
    """Mask expression only and return a full explicit mask channel.

    The returned metadata is a value-preserving copy (or ``None``).  This makes
    accidental in-place metadata masking visible in tests and call sites.
    """

    try:
        import torch
    except ImportError:  # pragma: no cover - exercised in minimal environments
        torch = None

    if torch is not None and torch.is_tensor(expression):
        gene_mask = torch.as_tensor(mask, dtype=torch.bool, device=expression.device)
        if tuple(gene_mask.shape) != tuple(expression.shape):
            raise ValueError("mask and expression must have the same shape")
        masked = expression.clone()
        masked.masked_fill_(gene_mask, float(fill_value))
        mask_channel = gene_mask.to(dtype=expression.dtype)
        if metadata is None:
            metadata_copy = None
        elif torch.is_tensor(metadata):
            metadata_copy = metadata.clone()
        else:
            metadata_copy = copy.deepcopy(metadata)
        return MaskedInputs(masked, mask_channel, metadata_copy)

    array = np.asarray(expression)
    gene_mask_np = np.asarray(mask, dtype=bool)
    if gene_mask_np.shape != array.shape:
        raise ValueError("mask and expression must have the same shape")
    masked_np = np.array(array, copy=True)
    masked_np[gene_mask_np] = fill_value
    metadata_copy = None if metadata is None else copy.deepcopy(metadata)
    return MaskedInputs(masked_np, gene_mask_np.astype(np.float32), metadata_copy)


def _array_checksum(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    header = _canonical_json_bytes(
        {"shape": list(array.shape), "dtype": array.dtype.str}
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _manifest_content_checksum(manifest: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(manifest))
    core.pop("bundle_checksum", None)
    core.pop("bundle_id", None)
    return hashlib.sha256(_canonical_json_bytes(core)).hexdigest()


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return clean or "split"


@dataclass(frozen=True)
class FixedMaskBundle:
    """Fixed validation/test mask replicates plus a content manifest."""

    masks: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        masks: dict[str, np.ndarray] = {}
        for key, value in self.masks.items():
            array = np.asarray(value)
            if array.ndim != 2 or array.dtype != np.bool_:
                raise TypeError(f"bundle mask {key!r} is not boolean [nodes, genes]")
            array = np.ascontiguousarray(array.copy())
            array.flags.writeable = False
            masks[str(key)] = array
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "manifest", copy.deepcopy(dict(self.manifest)))
        self.validate()

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def checksum(self) -> str:
        return str(self.manifest["bundle_checksum"])

    def validate(self) -> None:
        manifest = self.manifest
        if manifest.get("format_version") != MASK_BUNDLE_FORMAT_VERSION:
            raise ValueError("unsupported fixed-mask bundle format")
        expected_checksum = _manifest_content_checksum(manifest)
        if manifest.get("bundle_checksum") != expected_checksum:
            raise ValueError("fixed-mask manifest content checksum mismatch")
        if manifest.get("bundle_id") != expected_checksum[:16]:
            raise ValueError("fixed-mask bundle ID does not match checksum")
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError("fixed-mask manifest entries must be a list")
        entry_keys = [str(entry["entry_id"]) for entry in entries]
        if len(entry_keys) != len(set(entry_keys)):
            raise ValueError("fixed-mask manifest contains duplicate entry IDs")
        expected_keys = set(entry_keys)
        if expected_keys != set(self.masks):
            raise ValueError("fixed-mask manifest and array keys differ")
        n_genes = int(manifest["n_genes"])
        splits = manifest.get("splits")
        if not isinstance(splits, Mapping) or not splits:
            raise ValueError("fixed-mask manifest splits must be a non-empty mapping")
        for entry in entries:
            key = str(entry["entry_id"])
            mask = self.masks[key]
            split = str(entry.get("split"))
            if split not in splits:
                raise ValueError(f"unknown split for fixed mask {key}")
            if list(mask.shape) != list(entry["shape"]) or mask.shape[1] != n_genes:
                raise ValueError(f"shape mismatch for fixed mask {key}")
            if mask.shape[0] != int(splits[split]["n_nodes"]):
                raise ValueError(f"split node-count mismatch for fixed mask {key}")
            if _array_checksum(mask) != entry["mask_checksum"]:
                raise ValueError(f"content checksum mismatch for fixed mask {key}")
            summary = entry.get("summary", {})
            if list(summary.get("shape", [])) != list(mask.shape):
                raise ValueError(f"summary shape mismatch for fixed mask {key}")
            if int(summary.get("n_masked_entries", -1)) != int(mask.sum()):
                raise ValueError(f"summary masked-count mismatch for fixed mask {key}")
            if int(summary.get("n_selected_nodes", -1)) != int(
                mask.any(axis=1).sum()
            ):
                raise ValueError(f"summary selected-node mismatch for fixed mask {key}")

    def get(self, split: str, mode_or_spec_id: str, replicate: int = 0) -> np.ndarray:
        """Retrieve one fixed mask, rejecting ambiguous repeated specifications."""

        canonical_mode: str | None
        try:
            canonical_mode = _canonical_mode(mode_or_spec_id)
        except ValueError:
            canonical_mode = None
        matches: list[str] = []
        for entry in self.manifest["entries"]:
            spec = entry["spec"]
            if (
                entry["split"] == split
                and int(entry["replicate"]) == int(replicate)
                and (
                    entry["spec_id"] == mode_or_spec_id
                    or (canonical_mode is not None and spec["mode"] == canonical_mode)
                )
            ):
                matches.append(entry["entry_id"])
        if not matches:
            raise KeyError(
                f"No fixed mask for split={split!r}, spec={mode_or_spec_id!r}, "
                f"replicate={replicate}"
            )
        if len(matches) > 1:
            raise KeyError(
                f"Mask mode {mode_or_spec_id!r} is ambiguous; use a manifest spec_id"
            )
        return self.masks[matches[0]]


def create_fixed_mask_bundle(
    coordinates_um: Mapping[str, Any],
    n_genes: int,
    specs: Sequence[MaskSpec | Mapping[str, Any]] | MaskSpec | None = None,
    *,
    eligible_nodes: Mapping[str, Any] | None = None,
    replicates: int = 3,
    base_seed: int = 0,
) -> FixedMaskBundle:
    """Create paired fixed mask replicates for named validation/test splits.

    Coordinates are used to create block masks and to checksum the evaluation
    population, but are not written to the mask artifact.
    """

    if not isinstance(coordinates_um, Mapping) or not coordinates_um:
        raise TypeError("coordinates_um must map split names to coordinate arrays")
    if not all(isinstance(name, str) and name for name in coordinates_um):
        raise TypeError("fixed-mask split names must be non-empty strings")
    n_genes = int(n_genes)
    replicates = int(replicates)
    if n_genes <= 0 or replicates <= 0:
        raise ValueError("n_genes and replicates must be positive")
    if specs is None:
        normalised_specs = [
            MaskSpec("partial"),
            MaskSpec("node"),
            MaskSpec("block"),
        ]
    elif isinstance(specs, MaskSpec):
        normalised_specs = [specs]
    else:
        normalised_specs = [
            spec if isinstance(spec, MaskSpec) else MaskSpec(**dict(spec))
            for spec in specs
        ]
    if not normalised_specs:
        raise ValueError("at least one mask specification is required")

    masks: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    split_records: dict[str, Any] = {}
    for split in sorted(coordinates_um):
        coordinates = _normalise_coordinates(coordinates_um[split])
        if eligible_nodes is None:
            eligible = np.ones(coordinates.shape[0], dtype=bool)
        else:
            if split not in eligible_nodes:
                raise KeyError(f"eligible_nodes is missing split {split!r}")
            eligible = _normalise_eligible(eligible_nodes[split], coordinates.shape[0])
        split_records[split] = {
            "n_nodes": int(coordinates.shape[0]),
            "n_eligible_nodes": int(eligible.sum()),
            "coordinates_checksum": _array_checksum(coordinates),
            "eligible_nodes_checksum": _array_checksum(eligible),
        }
        for spec_index, spec in enumerate(normalised_specs):
            for replicate in range(replicates):
                mask_seed = derive_mask_seed(
                    base_seed,
                    "fixed-evaluation-mask",
                    split,
                    spec.spec_id,
                    replicate,
                )
                batch = generate_mask(
                    spec,
                    n_genes,
                    coordinates,
                    eligible_nodes=eligible,
                    seed=mask_seed,
                )
                entry_id = (
                    f"{_safe_id(split)}__{spec_index:02d}-{spec.spec_id}"
                    f"__r{replicate:03d}"
                )
                masks[entry_id] = batch.mask
                entries.append(
                    {
                        "entry_id": entry_id,
                        "split": split,
                        "spec_index": spec_index,
                        "spec_id": spec.spec_id,
                        "spec": spec.to_dict(),
                        "replicate": replicate,
                        "seed": mask_seed,
                        "shape": list(batch.mask.shape),
                        "mask_checksum": _array_checksum(batch.mask),
                        "summary": batch.summary(),
                    }
                )

    manifest: dict[str, Any] = {
        "format_version": MASK_BUNDLE_FORMAT_VERSION,
        "base_seed": int(base_seed),
        "n_genes": n_genes,
        "replicates": replicates,
        "splits": split_records,
        "entries": entries,
        "seed_contract": (
            "Mask seeds derive only from base_seed, split, specification, and "
            "replicate; model seeds are excluded."
        ),
    }
    checksum = _manifest_content_checksum(manifest)
    manifest["bundle_checksum"] = checksum
    manifest["bundle_id"] = checksum[:16]
    return FixedMaskBundle(masks=masks, manifest=manifest)


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_fixed_mask_bundle(
    bundle: FixedMaskBundle,
    path: str | os.PathLike[str],
) -> Path:
    """Atomically save an immutable bundle directory and file checksums."""

    if not isinstance(bundle, FixedMaskBundle):
        raise TypeError("bundle must be a FixedMaskBundle")
    bundle.validate()
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite fixed-mask artifact {destination}; "
            "use a new immutable path"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        manifest_path = temporary / "manifest.json"
        masks_path = temporary / "masks.npz"
        manifest_path.write_text(
            json.dumps(
                bundle.manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            masks_path,
            **{key: bundle.masks[key] for key in sorted(bundle.masks)},
        )
        file_checksums = {
            "manifest.json": _file_checksum(manifest_path),
            "masks.npz": _file_checksum(masks_path),
        }
        (temporary / "checksums.sha256").write_text(
            "".join(
                f"{checksum}  {name}\n"
                for name, checksum in sorted(file_checksums.items())
            ),
            encoding="ascii",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _read_checksum_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (manifest\.json|masks\.npz)", raw_line)
        if match is None:
            raise ValueError("invalid checksums.sha256 line")
        checksum, name = match.groups()
        if name in result:
            raise ValueError(f"duplicate checksum entry for {name}")
        result[name] = checksum
    if set(result) != {"manifest.json", "masks.npz"}:
        raise ValueError("checksums.sha256 does not cover both bundle files")
    return result


def load_fixed_mask_bundle(path: str | os.PathLike[str]) -> FixedMaskBundle:
    """Load a fixed-mask bundle after verifying file and content checksums."""

    source = Path(path)
    if not source.is_dir():
        raise FileNotFoundError(f"fixed-mask bundle directory not found: {source}")
    required = {
        "manifest.json": source / "manifest.json",
        "masks.npz": source / "masks.npz",
        "checksums.sha256": source / "checksums.sha256",
    }
    missing = [name for name, file_path in required.items() if not file_path.is_file()]
    if missing:
        raise FileNotFoundError(f"fixed-mask bundle is missing: {', '.join(missing)}")
    expected = _read_checksum_file(required["checksums.sha256"])
    for name in ("manifest.json", "masks.npz"):
        actual = _file_checksum(required[name])
        if actual != expected[name]:
            raise ValueError(f"file checksum mismatch for {name}")

    manifest = json.loads(required["manifest.json"].read_text(encoding="utf-8"))
    with np.load(required["masks.npz"], allow_pickle=False) as archive:
        masks = {key: np.asarray(archive[key]) for key in archive.files}
    return FixedMaskBundle(masks=masks, manifest=manifest)


# Short aliases are useful at orchestration call sites while the longer names
# keep the artifact's fixed-evaluation purpose explicit in reports.
save_mask_bundle = save_fixed_mask_bundle
load_mask_bundle = load_fixed_mask_bundle

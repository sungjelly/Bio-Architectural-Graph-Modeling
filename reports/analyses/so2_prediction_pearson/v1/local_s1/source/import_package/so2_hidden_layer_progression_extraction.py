"""Exact, resumable h0--h3 extraction for the locked SO2 four-block model.

The fourth captured graph state is deliberately not duplicated.  It is checked
to strict numerical tolerance against the already verified per-core ``hL``
artifacts produced by ``so2_hl_clustering``; exact equality is recorded as a
separate diagnostic.  These tensors are descriptive model states and carry no
cell-type or mechanistic interpretation by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from .fingerprints import sha256_file
from .relative_qkv_embedding_clustering import (
    _array_sha256,
    _atomic_write_json,
    _file_record,
    _read_json,
    _receipt_with_self_hash,
    _tensor_sha256,
    _verify_self_hash,
    _write_deterministic_npz,
)
from .relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from .so2_hl_clustering import (
    DEFAULT_CPU_THREADS,
    SO2ResolvedInputs,
    _expected_core_records,
    _verify_extraction_manifest as _verify_hl_extraction_manifest,
    load_contextual_core,
    load_so2_checkpoint_model,
)
from .so2_pooled_full_core import (
    EXPECTED_TOTAL_CELLS,
    SO2_ALIASES,
    SO2_CORE_NUMBERS,
)
from .so2_relative_graphs import load_so2_relative_qkv_batches


EXTRACTION_SCHEMA = "so2_14core_hidden_layer_progression_extraction_v1"
CORE_EXTRACTION_SCHEMA = "so2_14core_hidden_layer_progression_core_v1"
STORED_LAYER_NAMES = ("h0", "h1", "h2", "h3")
CAPTURED_LAYER_NAMES = (*STORED_LAYER_NAMES, "h4")
EXPECTED_GRAPH_STEPS = 4
REFERENCE_HL_ANALYSIS = "so2_14core_contextual_embedding_clustering"
H4_HL_ABSOLUTE_TOLERANCE = 2e-6
H4_HL_RELATIVE_TOLERANCE = 1e-6


class SO2HiddenLayerProgressionExtractionError(ValueError):
    """Raised when the locked hidden-layer extraction contract is violated."""


@dataclass(frozen=True, slots=True)
class SO2HiddenLayerCore:
    """Cell-aligned stored states for one complete SO2 core.

    ``h4`` is intentionally absent: use the verified source hL artifact named
    in the associated receipt.
    """

    alias: str
    core_number: int
    cell_index: np.ndarray = field(repr=False)
    coordinates_um: np.ndarray = field(repr=False)
    h0: np.ndarray = field(repr=False)
    h1: np.ndarray = field(repr=False)
    h2: np.ndarray = field(repr=False)
    h3: np.ndarray = field(repr=False)

    @property
    def n_cells(self) -> int:
        return int(len(self.cell_index))

    def layer(self, name: str) -> np.ndarray:
        """Return one stored layer by its exact h0--h3 name."""

        if name not in STORED_LAYER_NAMES:
            raise SO2HiddenLayerProgressionExtractionError(
                f"Stored layer must be one of {STORED_LAYER_NAMES}; observed {name!r}."
            )
        return getattr(self, name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_cpu_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise SO2HiddenLayerProgressionExtractionError(
            "SO2 hidden-layer progression extraction is CPU-only."
        )
    return torch.device("cpu")


def default_verified_hl_output_root(inputs: SO2ResolvedInputs) -> Path:
    """Return the report path containing the checksum-verified SO2 hL arrays."""

    return (
        inputs.project_root
        / "reports"
        / "analyses"
        / REFERENCE_HL_ANALYSIS
        / inputs.run_id
    )


@torch.inference_mode()
def extract_full_hidden_layer_progression(
    model: ReceiverChunkedRelativeGeometryQKVGraphTransformer,
    *,
    input_expression: torch.Tensor,
    gene_mask: torch.Tensor,
    edge_index: torch.Tensor,
    relative_geometry: torch.Tensor,
    node_covariates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture full-node h0, h1, h2, h3, and h4 in one model forward.

    h0 is cloned by a forward hook on the exact encoder invocation used by the
    graph forward.  h1--h4 come from ``graph_step_embeddings`` and therefore
    remain full-node tensors even though the decoder receives zero target rows.
    """

    if model.training or any(module.training for module in model.modules()):
        raise SO2HiddenLayerProgressionExtractionError(
            "Hidden-layer extraction requires model.eval()."
        )
    tensors = {
        "input_expression": input_expression,
        "gene_mask": gene_mask,
        "edge_index": edge_index,
        "relative_geometry": relative_geometry,
        "node_covariates": node_covariates,
    }
    if any(value.device.type != "cpu" for value in tensors.values()):
        raise SO2HiddenLayerProgressionExtractionError(
            "Expression, mask, metadata, graph, and geometry must remain on CPU."
        )
    if gene_mask.dtype is not torch.bool or gene_mask.shape != input_expression.shape:
        raise SO2HiddenLayerProgressionExtractionError(
            "Extraction mask shape or dtype is invalid."
        )
    if bool(torch.any(gene_mask).item()):
        raise SO2HiddenLayerProgressionExtractionError(
            "Extraction requires an all-zero gene mask."
        )
    if int(model.graph_layers) != EXPECTED_GRAPH_STEPS or len(model.blocks) != (
        EXPECTED_GRAPH_STEPS
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            "Extraction requires the independently parameterized four-block model."
        )

    captured_h0: list[torch.Tensor] = []

    def capture_encoder_output(
        _module: torch.nn.Module,
        _arguments: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise SO2HiddenLayerProgressionExtractionError(
                "NodeEncoder did not return a tensor."
            )
        captured_h0.append(output.detach().clone())

    hook = model.encoder.register_forward_hook(capture_encoder_output)
    try:
        empty_targets = torch.empty((0,), dtype=torch.long, device="cpu")
        output = model(
            input_expression=input_expression,
            gene_mask=gene_mask,
            edge_index=edge_index,
            relative_geometry=relative_geometry,
            node_covariates=node_covariates,
            target_nodes=empty_targets,
            return_graph_step_embeddings=True,
        )
    finally:
        hook.remove()

    if len(captured_h0) != 1:
        raise SO2HiddenLayerProgressionExtractionError(
            "The one-forward extraction did not invoke NodeEncoder exactly once."
        )
    graph_steps = output.graph_step_embeddings
    if graph_steps is None or len(graph_steps) != EXPECTED_GRAPH_STEPS:
        raise SO2HiddenLayerProgressionExtractionError(
            "The model did not return all four full-node graph-step states."
        )
    if output.prediction.shape != (
        0,
        model.num_genes,
    ) or output.node_embedding.shape != (0, model.hidden_dim):
        raise SO2HiddenLayerProgressionExtractionError(
            "The decoder target selection was not empty."
        )
    if output.full_node_embedding is None or not torch.equal(
        graph_steps[-1], output.full_node_embedding
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            "Captured h4 differs from the public full-node hL tensor."
        )

    states = (captured_h0[0], *graph_steps)
    expected_shape = (int(input_expression.shape[0]), int(model.hidden_dim))
    for name, state in zip(CAPTURED_LAYER_NAMES, states, strict=True):
        if state.shape != expected_shape:
            raise SO2HiddenLayerProgressionExtractionError(
                f"Captured {name} shape is invalid: {tuple(state.shape)}."
            )
        if state.device.type != "cpu" or not bool(torch.isfinite(state).all().item()):
            raise SO2HiddenLayerProgressionExtractionError(
                f"Captured {name} must be finite and CPU-resident."
            )
    return states


def _core_embedding_path(output_root: Path, core_number: int) -> Path:
    return output_root / "embeddings" / f"core_{int(core_number)}_hidden_layers.npz"


def _core_receipt_path(output_root: Path, core_number: int) -> Path:
    return output_root / "embeddings" / f"core_{int(core_number)}_receipt.json"


def _validate_hidden_layer_arrays(
    *,
    alias: str,
    core_number: int,
    arrays: Mapping[str, np.ndarray],
    expected_cells: int,
    hidden_dim: int,
) -> SO2HiddenLayerCore:
    expected_names = {
        "cell_index",
        "core_number",
        "coordinates_um",
        *STORED_LAYER_NAMES,
    }
    if set(arrays) != expected_names:
        raise SO2HiddenLayerProgressionExtractionError(
            f"Unexpected hidden-layer artifact schema for {alias}."
        )
    cell_index = np.asarray(arrays["cell_index"])
    number = np.asarray(arrays["core_number"])
    coordinates = np.asarray(arrays["coordinates_um"])
    if cell_index.dtype != np.dtype(np.int64) or not np.array_equal(
        cell_index, np.arange(expected_cells, dtype=np.int64)
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            f"Cell order changed for {alias}."
        )
    if (
        number.dtype != np.dtype(np.int16)
        or number.size != 1
        or int(number.reshape(-1)[0]) != int(core_number)
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            f"Core number changed for {alias}."
        )
    if coordinates.dtype != np.dtype(np.float64) or coordinates.shape != (
        expected_cells,
        2,
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            f"Coordinate schema changed for {alias}."
        )
    layers: dict[str, np.ndarray] = {}
    for name in STORED_LAYER_NAMES:
        layer = np.asarray(arrays[name])
        if layer.dtype != np.dtype(np.float32) or layer.shape != (
            expected_cells,
            hidden_dim,
        ):
            raise SO2HiddenLayerProgressionExtractionError(
                f"Stored {name} schema changed for {alias}."
            )
        layers[name] = np.ascontiguousarray(layer)
    if not np.isfinite(coordinates).all() or any(
        not np.isfinite(layer).all() for layer in layers.values()
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            f"Non-finite hidden-layer values exist for {alias}."
        )
    return SO2HiddenLayerCore(
        alias=str(alias),
        core_number=int(core_number),
        cell_index=np.ascontiguousarray(cell_index),
        coordinates_um=np.ascontiguousarray(coordinates),
        h0=layers["h0"],
        h1=layers["h1"],
        h2=layers["h2"],
        h3=layers["h3"],
    )


def load_hidden_layer_core(
    path: Path,
    *,
    alias: str,
    core_number: int,
    expected_cells: int,
    hidden_dim: int,
) -> SO2HiddenLayerCore:
    """Load and strictly validate one ``core_<N>_hidden_layers.npz`` file."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise SO2HiddenLayerProgressionExtractionError(
            f"Cannot load hidden-layer artifact: {path}"
        ) from exc
    return _validate_hidden_layer_arrays(
        alias=alias,
        core_number=core_number,
        arrays=arrays,
        expected_cells=expected_cells,
        hidden_dim=hidden_dim,
    )


def _load_verified_hl_extraction(
    *,
    inputs: SO2ResolvedInputs,
    verified_hl_output_root: Path | None,
) -> tuple[Path, Mapping[str, Any], str]:
    root = (
        default_verified_hl_output_root(inputs)
        if verified_hl_output_root is None
        else Path(verified_hl_output_root)
    ).resolve(strict=True)
    manifest_path = root / "embeddings" / "extraction_manifest.json"
    receipt = _read_json(manifest_path, label="verified SO2 hL extraction manifest")
    _verify_hl_extraction_manifest(output_root=root, receipt=receipt, inputs=inputs)
    return root, receipt, sha256_file(manifest_path)


def _reference_hl_record(
    receipt: Mapping[str, Any], *, alias: str, core_number: int
) -> Mapping[str, Any]:
    records = receipt.get("cores")
    if not isinstance(records, list):
        raise SO2HiddenLayerProgressionExtractionError(
            "Verified hL extraction manifest lacks core records."
        )
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("alias")) == alias
        and int(record.get("core_number", -1)) == int(core_number)
    ]
    if len(matches) != 1:
        raise SO2HiddenLayerProgressionExtractionError(
            f"Verified hL source is ambiguous or missing for {alias}."
        )
    return matches[0]


def _layer_array_receipts(core: SO2HiddenLayerCore) -> dict[str, Any]:
    values: dict[str, np.ndarray] = {
        "cell_index": core.cell_index,
        "core_number": np.ascontiguousarray(
            np.asarray(core.core_number, dtype=np.int16)
        ),
        "coordinates_um": core.coordinates_um,
        **{name: core.layer(name) for name in STORED_LAYER_NAMES},
    }
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "array_sha256": _array_sha256(name, value),
        }
        for name, value in values.items()
    }


def _compare_captured_h4_to_verified_hl(
    captured_h4: np.ndarray, verified_hl: np.ndarray
) -> dict[str, Any]:
    """Return a strict replay receipt or fail when h4 exceeds tolerance."""

    captured = np.asarray(captured_h4)
    reference = np.asarray(verified_hl)
    same_shape = captured.shape == reference.shape
    valid_schema = (
        same_shape
        and captured.dtype == np.dtype(np.float32)
        and reference.dtype == np.dtype(np.float32)
        and np.isfinite(captured).all()
        and np.isfinite(reference).all()
    )
    differences = (
        np.abs(captured.astype(np.float64) - reference.astype(np.float64))
        if valid_schema
        else np.asarray([float("inf")], dtype=np.float64)
    )
    tolerance = (
        H4_HL_ABSOLUTE_TOLERANCE
        + H4_HL_RELATIVE_TOLERANCE * np.abs(reference.astype(np.float64))
        if valid_schema
        else np.asarray([1.0], dtype=np.float64)
    )
    maximum_difference = float(np.max(differences, initial=0.0))
    mean_difference = float(np.mean(differences))
    maximum_scaled_ratio = float(np.max(differences / tolerance, initial=0.0))
    equivalent = bool(
        valid_schema
        and np.allclose(
            captured,
            reference,
            rtol=H4_HL_RELATIVE_TOLERANCE,
            atol=H4_HL_ABSOLUTE_TOLERANCE,
            equal_nan=False,
        )
    )
    if not equivalent:
        raise SO2HiddenLayerProgressionExtractionError(
            "Captured h4 differs from verified hL beyond tolerance; "
            f"maximum absolute difference={maximum_difference}."
        )
    return {
        "source_hL_array_sha256": _array_sha256("hL", reference),
        "captured_h4_array_sha256": _array_sha256("h4", captured),
        "captured_h4_bitwise_equal": bool(np.array_equal(captured, reference)),
        "captured_h4_allclose": True,
        "maximum_absolute_difference": maximum_difference,
        "mean_absolute_difference": mean_difference,
        "maximum_scaled_tolerance_ratio": maximum_scaled_ratio,
        "absolute_tolerance": H4_HL_ABSOLUTE_TOLERANCE,
        "relative_tolerance": H4_HL_RELATIVE_TOLERANCE,
        "captured_h4_persisted": False,
        "same_process_h4_equals_full_node_embedding_exact": True,
    }


def _verify_core_extraction_receipt(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    inputs: SO2ResolvedInputs,
    expected: Mapping[str, Any],
    hidden_dim: int,
    verified_hl_output_root: Path,
    verified_hl_extraction: Mapping[str, Any],
    verified_hl_manifest_file_sha256: str,
) -> None:
    alias = str(expected["alias"])
    core_number = int(expected["core_number"])
    expected_cells = int(expected["cell_count"])
    _verify_self_hash(receipt, label="SO2 hidden-layer per-core receipt")
    if any(
        (
            receipt.get("schema") != CORE_EXTRACTION_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != inputs.run_id,
            receipt.get("checkpoint_sha256") != inputs.checkpoint_sha256,
            receipt.get("alias") != alias,
            int(receipt.get("core_number", -1)) != core_number,
            int(receipt.get("cell_count", -1)) != expected_cells,
            int(receipt.get("hidden_dim", -1)) != int(hidden_dim),
            tuple(receipt.get("stored_layers", ())) != STORED_LAYER_NAMES,
            tuple(receipt.get("captured_layers", ())) != CAPTURED_LAYER_NAMES,
            receipt.get("source_core") != dict(expected),
        )
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            f"Hidden-layer receipt identity changed for {alias}."
        )
    relative = str(receipt.get("embedding_file", ""))
    expected_relative = _core_embedding_path(output_root, core_number).relative_to(
        output_root
    ).as_posix()
    if relative != expected_relative:
        raise SO2HiddenLayerProgressionExtractionError(
            f"Hidden-layer output path changed for {alias}."
        )
    path = output_root / relative
    file_record = receipt.get("file")
    if not isinstance(file_record, Mapping) or not path.is_file():
        raise SO2HiddenLayerProgressionExtractionError(
            f"Hidden-layer output is missing for {alias}."
        )
    if _file_record(path) != dict(file_record):
        raise SO2HiddenLayerProgressionExtractionError(
            f"Hidden-layer file checksum changed for {alias}."
        )
    core = load_hidden_layer_core(
        path,
        alias=alias,
        core_number=core_number,
        expected_cells=expected_cells,
        hidden_dim=hidden_dim,
    )
    if receipt.get("arrays") != _layer_array_receipts(core):
        raise SO2HiddenLayerProgressionExtractionError(
            f"Hidden-layer array checksum changed for {alias}."
        )

    source_record = _reference_hl_record(
        verified_hl_extraction, alias=alias, core_number=core_number
    )
    source_file = verified_hl_output_root / str(source_record.get("embedding_file", ""))
    source_core = load_contextual_core(
        source_file,
        alias=alias,
        core_number=core_number,
        expected_cells=expected_cells,
    )
    if not np.array_equal(
        source_core.cell_index, core.cell_index
    ) or not np.array_equal(source_core.coordinates_um, core.coordinates_um):
        raise SO2HiddenLayerProgressionExtractionError(
            f"Verified hL rows or coordinates do not align for {alias}."
        )
    source_hl = source_core.hL
    source_hl_sha = _array_sha256("hL", source_hl)
    reference = receipt.get("reference_hL")
    numeric_fields = (
        float(reference.get("maximum_absolute_difference", float("nan")))
        if isinstance(reference, Mapping)
        else float("nan"),
        float(reference.get("mean_absolute_difference", float("nan")))
        if isinstance(reference, Mapping)
        else float("nan"),
        float(reference.get("maximum_scaled_tolerance_ratio", float("nan")))
        if isinstance(reference, Mapping)
        else float("nan"),
    )
    if not isinstance(reference, Mapping) or any(
        (
            reference.get("source_extraction_manifest_file_sha256")
            != verified_hl_manifest_file_sha256,
            reference.get("source_embedding_file")
            != str(source_record.get("embedding_file", "")),
            reference.get("source_embedding_file_record")
            != _file_record(source_file),
            reference.get("source_hL_array_sha256") != source_hl_sha,
            reference.get("source_hL_array_sha256")
            != source_record.get("hL_array_sha256"),
            reference.get("source_coordinates_array_sha256")
            != _array_sha256("coordinates_um", source_core.coordinates_um),
            reference.get("source_cell_index_array_sha256")
            != _array_sha256("cell_index", source_core.cell_index),
            not isinstance(reference.get("captured_h4_array_sha256"), str),
            reference.get("captured_h4_allclose") is not True,
            not isinstance(reference.get("captured_h4_bitwise_equal"), bool),
            float(reference.get("absolute_tolerance", float("nan")))
            != H4_HL_ABSOLUTE_TOLERANCE,
            float(reference.get("relative_tolerance", float("nan")))
            != H4_HL_RELATIVE_TOLERANCE,
            not all(math.isfinite(value) and value >= 0.0 for value in numeric_fields),
            numeric_fields[2] > 1.0,
            reference.get("captured_h4_persisted") is not False,
            reference.get("same_process_h4_equals_full_node_embedding_exact")
            is not True,
        )
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            f"Captured h4 reference changed for {alias}."
        )


def _verify_extraction_manifest(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    inputs: SO2ResolvedInputs,
    hidden_dim: int,
    verified_hl_output_root: Path,
    verified_hl_extraction: Mapping[str, Any],
    verified_hl_manifest_file_sha256: str,
) -> None:
    _verify_self_hash(receipt, label="SO2 hidden-layer extraction manifest")
    if any(
        (
            receipt.get("schema") != EXTRACTION_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != inputs.run_id,
            receipt.get("checkpoint_sha256") != inputs.checkpoint_sha256,
            tuple(receipt.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(receipt.get("embedding_dimension", -1)) != int(hidden_dim),
            tuple(receipt.get("stored_layers", ())) != STORED_LAYER_NAMES,
            tuple(receipt.get("captured_layers", ())) != CAPTURED_LAYER_NAMES,
            receipt.get("device") != "cpu",
            receipt.get("source_hL_extraction_manifest_file_sha256")
            != verified_hl_manifest_file_sha256,
        )
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            "SO2 hidden-layer extraction manifest identity is invalid."
        )
    records = receipt.get("cores")
    expected_records = _expected_core_records(inputs)
    if not isinstance(records, list) or len(records) != len(expected_records):
        raise SO2HiddenLayerProgressionExtractionError(
            "SO2 hidden-layer extraction manifest is incomplete."
        )
    for observed, expected in zip(records, expected_records, strict=True):
        if not isinstance(observed, Mapping):
            raise SO2HiddenLayerProgressionExtractionError(
                "SO2 hidden-layer core receipt is malformed."
            )
        _verify_core_extraction_receipt(
            output_root=output_root,
            receipt=observed,
            inputs=inputs,
            expected=expected,
            hidden_dim=hidden_dim,
            verified_hl_output_root=verified_hl_output_root,
            verified_hl_extraction=verified_hl_extraction,
            verified_hl_manifest_file_sha256=verified_hl_manifest_file_sha256,
        )


def extract_hidden_layer_progression(
    *,
    inputs: SO2ResolvedInputs,
    output_root: Path,
    verified_hl_output_root: Path | None = None,
    device: str | torch.device = "cpu",
    cpu_threads: int = DEFAULT_CPU_THREADS,
) -> Mapping[str, Any]:
    """Extract or checksum-verify h0--h3 for all 14 complete SO2 cores.

    The source model, input manifests, and checkpoint have already been locked
    by ``resolve_so2_analysis_inputs``.  This stage performs exactly one model
    forward for each core that lacks a valid per-core receipt.
    """

    validate_cpu_device(device)
    if isinstance(cpu_threads, bool) or int(cpu_threads) <= 0:
        raise SO2HiddenLayerProgressionExtractionError(
            "cpu_threads must be a positive integer."
        )
    construction = inputs.checkpoint_payload.get("model_construction")
    if not isinstance(construction, Mapping) or any(
        (
            construction.get("class")
            != "ReceiverChunkedRelativeGeometryQKVGraphTransformer",
            int(construction.get("graph_layers", -1)) != EXPECTED_GRAPH_STEPS,
        )
    ):
        raise SO2HiddenLayerProgressionExtractionError(
            "Checkpoint is not the locked independently parameterized four-block model."
        )
    hidden_dim = int(construction.get("hidden_dim", -1))
    if hidden_dim <= 0:
        raise SO2HiddenLayerProgressionExtractionError(
            "Checkpoint hidden dimension is invalid."
        )

    (
        hl_root,
        hl_extraction,
        hl_manifest_file_sha,
    ) = _load_verified_hl_extraction(
        inputs=inputs, verified_hl_output_root=verified_hl_output_root
    )
    source_cpu_threads = int(hl_extraction.get("cpu_threads", -1))
    if source_cpu_threads <= 0 or int(cpu_threads) != source_cpu_threads:
        raise SO2HiddenLayerProgressionExtractionError(
            "CPU thread count must match the verified hL extraction for matched "
            "numerical replay."
        )

    output_root = Path(output_root)
    manifest_path = output_root / "embeddings" / "extraction_manifest.json"
    if manifest_path.is_file():
        receipt = _read_json(
            manifest_path, label="SO2 hidden-layer extraction manifest"
        )
        _verify_extraction_manifest(
            output_root=output_root,
            receipt=receipt,
            inputs=inputs,
            hidden_dim=hidden_dim,
            verified_hl_output_root=hl_root,
            verified_hl_extraction=hl_extraction,
            verified_hl_manifest_file_sha256=hl_manifest_file_sha,
        )
        return receipt

    expected_records = _expected_core_records(inputs)
    receipts_by_alias: dict[str, Mapping[str, Any]] = {}
    missing_aliases: set[str] = set()
    for expected in expected_records:
        alias = str(expected["alias"])
        core_number = int(expected["core_number"])
        file_path = _core_embedding_path(output_root, core_number)
        core_receipt_path = _core_receipt_path(output_root, core_number)
        if core_receipt_path.is_file():
            core_receipt = _read_json(
                core_receipt_path, label=f"{alias} hidden-layer extraction receipt"
            )
            _verify_core_extraction_receipt(
                output_root=output_root,
                receipt=core_receipt,
                inputs=inputs,
                expected=expected,
                hidden_dim=hidden_dim,
                verified_hl_output_root=hl_root,
                verified_hl_extraction=hl_extraction,
                verified_hl_manifest_file_sha256=hl_manifest_file_sha,
            )
            receipts_by_alias[alias] = core_receipt
        elif file_path.exists():
            raise SO2HiddenLayerProgressionExtractionError(
                f"Unreceipted partial hidden-layer output exists for {alias}."
            )
        else:
            missing_aliases.add(alias)

    if missing_aliases:
        torch.set_num_threads(int(cpu_threads))
        batches = load_so2_relative_qkv_batches(
            cohort_dir=inputs.cohort_dir, graph_dir=inputs.graph_dir
        )
        if tuple(batch.alias for batch in batches) != SO2_ALIASES:
            raise SO2HiddenLayerProgressionExtractionError(
                "Loaded SO2 batches are missing or reordered."
            )
        batch_by_alias = {str(batch.alias): batch for batch in batches}
        model = load_so2_checkpoint_model(inputs)
        schedule = sorted(
            (
                expected
                for expected in expected_records
                if str(expected["alias"]) in missing_aliases
            ),
            key=lambda record: (int(record["cell_count"]), int(record["core_number"])),
        )
        for expected in schedule:
            alias = str(expected["alias"])
            core_number = int(expected["core_number"])
            expected_cells = int(expected["cell_count"])
            batch = batch_by_alias[alias]
            if int(batch.n_nodes) != expected_cells:
                raise SO2HiddenLayerProgressionExtractionError(
                    f"Loaded cell count changed for {alias}."
                )
            with np.load(
                inputs.cohort_dir / "cores" / f"{alias}.npz", allow_pickle=False
            ) as archive:
                coordinates = np.array(
                    archive["coordinates_um"], dtype=np.float64, copy=True
                )
            if coordinates.shape != (expected_cells, 2):
                raise SO2HiddenLayerProgressionExtractionError(
                    f"Coordinates do not align for {alias}."
                )
            expression = batch.target_expression.to(dtype=torch.float32, device="cpu")
            covariates = batch.node_covariates.to(dtype=torch.float32, device="cpu")
            gene_mask = torch.zeros_like(expression, dtype=torch.bool, device="cpu")
            source_tensors = {
                "target_expression": batch.target_expression,
                "node_covariates": batch.node_covariates,
                "edge_index": batch.edge_index,
                "relative_geometry": batch.relative_geometry,
            }
            source_hashes = {
                name: _tensor_sha256(name, tensor)
                for name, tensor in source_tensors.items()
            }
            started = time.monotonic()
            states = extract_full_hidden_layer_progression(
                model,
                input_expression=expression,
                gene_mask=gene_mask,
                edge_index=batch.edge_index,
                relative_geometry=batch.relative_geometry,
                node_covariates=covariates,
            )
            elapsed = time.monotonic() - started
            if source_hashes != {
                name: _tensor_sha256(name, tensor)
                for name, tensor in source_tensors.items()
            }:
                raise SO2HiddenLayerProgressionExtractionError(
                    f"Prepared inputs mutated during extraction for {alias}."
                )
            numpy_states = tuple(
                np.ascontiguousarray(state.detach().cpu().float().numpy())
                for state in states
            )
            reference_record = _reference_hl_record(
                hl_extraction, alias=alias, core_number=core_number
            )
            reference_path = hl_root / str(reference_record["embedding_file"])
            reference_core = load_contextual_core(
                reference_path,
                alias=alias,
                core_number=core_number,
                expected_cells=expected_cells,
            )
            if not np.array_equal(
                reference_core.cell_index,
                np.arange(expected_cells, dtype=np.int64),
            ) or not np.array_equal(reference_core.coordinates_um, coordinates):
                raise SO2HiddenLayerProgressionExtractionError(
                    f"Verified hL rows or coordinates do not align for {alias}."
                )
            reference_hl = reference_core.hL
            captured_h4 = numpy_states[-1]
            h4_comparison = _compare_captured_h4_to_verified_hl(
                captured_h4, reference_hl
            )

            arrays = {
                "cell_index": np.arange(expected_cells, dtype=np.int64),
                "core_number": np.asarray(core_number, dtype=np.int16),
                "coordinates_um": coordinates,
                **{
                    name: state
                    for name, state in zip(
                        STORED_LAYER_NAMES, numpy_states[:-1], strict=True
                    )
                },
            }
            core = _validate_hidden_layer_arrays(
                alias=alias,
                core_number=core_number,
                arrays=arrays,
                expected_cells=expected_cells,
                hidden_dim=hidden_dim,
            )
            output_file = _core_embedding_path(output_root, core_number)
            _write_deterministic_npz(
                output_file,
                {
                    "cell_index": core.cell_index,
                    "core_number": np.asarray(core_number, dtype=np.int16),
                    "coordinates_um": core.coordinates_um,
                    **{name: core.layer(name) for name in STORED_LAYER_NAMES},
                },
            )
            relative = output_file.relative_to(output_root).as_posix()
            core_receipt = _receipt_with_self_hash(
                {
                    "schema": CORE_EXTRACTION_SCHEMA,
                    "status": "complete",
                    "created_at": _utc_now(),
                    "run_id": inputs.run_id,
                    "checkpoint_sha256": inputs.checkpoint_sha256,
                    "alias": alias,
                    "core_number": core_number,
                    "cell_count": core.n_cells,
                    "hidden_dim": hidden_dim,
                    "stored_layers": list(STORED_LAYER_NAMES),
                    "captured_layers": list(CAPTURED_LAYER_NAMES),
                    "source_core": dict(expected),
                    "source_tensor_sha256": source_hashes,
                    "arrays": _layer_array_receipts(core),
                    "reference_hL": {
                        "source_extraction_manifest_file_sha256": hl_manifest_file_sha,
                        "source_embedding_file": str(
                            reference_record["embedding_file"]
                        ),
                        "source_embedding_file_record": _file_record(reference_path),
                        "source_coordinates_array_sha256": _array_sha256(
                            "coordinates_um", reference_core.coordinates_um
                        ),
                        "source_cell_index_array_sha256": _array_sha256(
                            "cell_index", reference_core.cell_index
                        ),
                        **h4_comparison,
                    },
                    "inference": {
                        "device": "cpu",
                        "cpu_threads": int(cpu_threads),
                        "model_eval": True,
                        "torch_inference_mode": True,
                        "model_forward_count": 1,
                        "encoder_forward_count": 1,
                        "decoder_target_rows": 0,
                        "gene_mask_nonzero_count": 0,
                        "complete_core": True,
                        "neighbor_sampling": False,
                        "graph_step_count": EXPECTED_GRAPH_STEPS,
                        "elapsed_seconds": float(elapsed),
                    },
                    "embedding_file": relative,
                    "file": _file_record(output_file),
                }
            )
            core_receipt_path = _core_receipt_path(output_root, core_number)
            _atomic_write_json(core_receipt_path, core_receipt)
            _verify_core_extraction_receipt(
                output_root=output_root,
                receipt=core_receipt,
                inputs=inputs,
                expected=expected,
                hidden_dim=hidden_dim,
                verified_hl_output_root=hl_root,
                verified_hl_extraction=hl_extraction,
                verified_hl_manifest_file_sha256=hl_manifest_file_sha,
            )
            receipts_by_alias[alias] = core_receipt
            del (
                states,
                numpy_states,
                captured_h4,
                reference_hl,
                reference_core,
                expression,
                covariates,
                gene_mask,
                core,
            )
            model.clear_edge_layout_cache()
            gc.collect()
        del model, batches
        gc.collect()

    core_receipts = [receipts_by_alias[alias] for alias in SO2_ALIASES]
    receipt = _receipt_with_self_hash(
        {
            "schema": EXTRACTION_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "run_id": inputs.run_id,
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "core_order": list(SO2_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "embedding_dimension": hidden_dim,
            "stored_layers": list(STORED_LAYER_NAMES),
            "captured_layers": list(CAPTURED_LAYER_NAMES),
            "h4_storage": "reuse_verified_hL_not_duplicated",
            "device": "cpu",
            "cpu_threads": int(cpu_threads),
            "one_complete_core_at_a_time": True,
            "inference_schedule": "ascending_cell_count_then_core_number",
            "stored_core_order": list(SO2_CORE_NUMBERS),
            "all_zero_extraction_masks": True,
            "model_forward_count_per_new_core": 1,
            "decoder_target_rows_per_core": 0,
            "source_hL_output_root": hl_root.as_posix(),
            "source_hL_extraction_manifest_file_sha256": hl_manifest_file_sha,
            "input_provenance": inputs.provenance,
            "cores": core_receipts,
        }
    )
    _atomic_write_json(manifest_path, receipt)
    _verify_extraction_manifest(
        output_root=output_root,
        receipt=receipt,
        inputs=inputs,
        hidden_dim=hidden_dim,
        verified_hl_output_root=hl_root,
        verified_hl_extraction=hl_extraction,
        verified_hl_manifest_file_sha256=hl_manifest_file_sha,
    )
    return receipt


__all__ = [
    "CAPTURED_LAYER_NAMES",
    "CORE_EXTRACTION_SCHEMA",
    "EXTRACTION_SCHEMA",
    "H4_HL_ABSOLUTE_TOLERANCE",
    "H4_HL_RELATIVE_TOLERANCE",
    "SO2HiddenLayerCore",
    "SO2HiddenLayerProgressionExtractionError",
    "STORED_LAYER_NAMES",
    "default_verified_hl_output_root",
    "extract_full_hidden_layer_progression",
    "extract_hidden_layer_progression",
    "load_hidden_layer_core",
    "validate_cpu_device",
]

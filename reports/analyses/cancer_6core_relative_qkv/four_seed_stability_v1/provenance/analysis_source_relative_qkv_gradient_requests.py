"""Locked, model-independent gradient probes for relative-QKV stability.

The probe table is selected only from checksum-verified prepared topology,
plotting coordinates, the ordered gene schema, and the deterministic held-in
mask.  No checkpoint, prediction, attention value, or gradient participates in
selection.  The resulting derivatives remain local model sensitivities; they
are not causal effects or independent biological replication.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .cancer_pooled_full_core import CANCER_ALIASES


CAMPAIGN_ID = "cmp_20260824_cancer_6core_relative_qkv_multiseed"
GRADIENT_REQUEST_SCHEMA = (
    "cancer_6core_relative_qkv_locked_gradient_requests_v1"
)
GRADIENT_PROTOCOL_SCHEMA = (
    "cancer_6core_relative_qkv_stability_analysis_protocol_v1"
)
GRADIENT_SELECTION_NAMESPACE = (
    "cancer-6core-relative-qkv-model-independent-gradient-probes-v1"
)
EXPECTED_ACTIVE_MODEL_SEEDS = (0, 1, 2, 3)
DEFERRED_MODEL_SEEDS = (4,)
FINAL_LAYER = -1
ATTENTION_HEAD = "mean"
RADIAL_SHELLS = (
    (0.0, 50.0, "(0,50]"),
    (50.0, 150.0, "(50,150]"),
    (150.0, 300.0, "(150,300]"),
    (300.0, 500.0, "(300,500]"),
)
EXPECTED_REQUEST_COUNT = len(CANCER_ALIASES) * len(RADIAL_SHELLS)
_EDGE_SCAN_CHUNK_SIZE = 1_000_000

REQUEST_CSV_FIELDS = (
    "request_schema",
    "request_id",
    "core_alias",
    "layer",
    "attention_head",
    "canonical_edge_id",
    "canonical_shell_candidate_index",
    "shell_candidate_count",
    "source_node",
    "receiver_node",
    "source_feature",
    "source_feature_index",
    "source_feature_name",
    "target_feature",
    "target_feature_index",
    "target_feature_name",
    "distance_um",
    "radial_shell_index",
    "radial_shell",
    "graph_sha256",
    "fixed_mask_seed",
    "fixed_mask_sha256",
    "assert_directed_edge",
    "assert_shell_membership",
    "assert_source_feature_observed",
    "assert_target_feature_masked",
    "assert_feature_indices_distinct",
)


class LockedGradientRequestError(RuntimeError):
    """Raised when a locked gradient request or its provenance drifts."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_index(*, role: str, alias: str, shell: str, count: int) -> int:
    if count <= 0:
        raise LockedGradientRequestError(
            f"Cannot select {role} from an empty candidate set."
        )
    payload = (
        f"{GRADIENT_SELECTION_NAMESPACE}\0{role}\0{alias}\0{shell}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % int(count)


def _cyclic_feature(
    valid: np.ndarray,
    *,
    role: str,
    alias: str,
    shell: str,
    excluded_index: int | None = None,
) -> int:
    candidate = np.asarray(valid, dtype=np.bool_)
    if candidate.ndim != 1 or len(candidate) == 0:
        raise LockedGradientRequestError("Feature eligibility must be one-dimensional.")
    start = _selection_index(
        role=role,
        alias=alias,
        shell=shell,
        count=len(candidate),
    )
    for offset in range(len(candidate)):
        index = (start + offset) % len(candidate)
        if bool(candidate[index]) and index != excluded_index:
            return int(index)
    raise LockedGradientRequestError(
        f"No eligible {role} exists for {alias} shell {shell}."
    )


@dataclass(frozen=True, slots=True)
class GradientRequestCoreInputs:
    """Prepared, checkpoint-independent inputs for one Cancer core."""

    alias: str
    edge_index: np.ndarray
    coordinates_um: np.ndarray
    gene_names: tuple[str, ...]
    fixed_mask: np.ndarray
    fixed_mask_seed: int
    fixed_mask_sha256: str
    graph_sha256: str


@dataclass(frozen=True, slots=True)
class LockedGradientRequest:
    """One checksum-bound source-gene to receiver-gene sensitivity probe."""

    request_id: str
    core_alias: str
    canonical_edge_id: int
    canonical_shell_candidate_index: int
    shell_candidate_count: int
    source_node: int
    receiver_node: int
    source_feature_index: int
    source_feature_name: str
    target_feature_index: int
    target_feature_name: str
    distance_um: float
    radial_shell_index: int
    radial_shell: str
    graph_sha256: str
    fixed_mask_seed: int
    fixed_mask_sha256: str
    layer: int = FINAL_LAYER
    attention_head: str = ATTENTION_HEAD

    def to_csv_row(self) -> dict[str, str]:
        source_index = str(int(self.source_feature_index))
        target_index = str(int(self.target_feature_index))
        return {
            "request_schema": GRADIENT_REQUEST_SCHEMA,
            "request_id": self.request_id,
            "core_alias": self.core_alias,
            "layer": str(int(self.layer)),
            "attention_head": self.attention_head,
            "canonical_edge_id": str(int(self.canonical_edge_id)),
            "canonical_shell_candidate_index": str(
                int(self.canonical_shell_candidate_index)
            ),
            "shell_candidate_count": str(int(self.shell_candidate_count)),
            "source_node": str(int(self.source_node)),
            "receiver_node": str(int(self.receiver_node)),
            # These two compatibility fields are consumed by the existing
            # selected-derivative CLI; explicit index/name fields remain beside them.
            "source_feature": source_index,
            "source_feature_index": source_index,
            "source_feature_name": self.source_feature_name,
            "target_feature": target_index,
            "target_feature_index": target_index,
            "target_feature_name": self.target_feature_name,
            "distance_um": format(float(self.distance_um), ".17g"),
            "radial_shell_index": str(int(self.radial_shell_index)),
            "radial_shell": self.radial_shell,
            "graph_sha256": self.graph_sha256,
            "fixed_mask_seed": str(int(self.fixed_mask_seed)),
            "fixed_mask_sha256": self.fixed_mask_sha256,
            "assert_directed_edge": "true",
            "assert_shell_membership": "true",
            "assert_source_feature_observed": "true",
            "assert_target_feature_masked": "true",
            "assert_feature_indices_distinct": "true",
        }


def load_prepared_gradient_request_inputs(
    *,
    cohort_dir: str | Path,
    graph_dir: str | Path,
) -> tuple[GradientRequestCoreInputs, ...]:
    """Load checksum-verified prepared inputs without loading any model state."""

    from .relative_qkv_post_training import (
        fixed_inference_mask,
        load_core_coordinates_and_genes,
        load_prepared_relative_qkv_batches,
    )

    cohort_root = Path(cohort_dir)
    graph_root = Path(graph_dir)
    batches = load_prepared_relative_qkv_batches(
        cohort_dir=cohort_root,
        graph_dir=graph_root,
    )
    try:
        graph_manifest = json.loads(
            (graph_root / "manifest.json").read_text(encoding="utf-8")
        )
        graph_records = {
            str(record["alias"]): record
            for record in graph_manifest["cores"]
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LockedGradientRequestError(
            "Cannot read the verified prepared graph manifest."
        ) from exc
    inputs: list[GradientRequestCoreInputs] = []
    for batch in batches:
        coordinates, genes = load_core_coordinates_and_genes(
            cohort_root,
            alias=batch.alias,
        )
        fixed_mask = fixed_inference_mask(batch)
        try:
            graph_sha256 = str(
                graph_records[batch.alias]["graph"]["checksums"]["graph_sha256"]
            )
        except (KeyError, TypeError) as exc:
            raise LockedGradientRequestError(
                f"Graph manifest lacks the semantic checksum for {batch.alias}."
            ) from exc
        inputs.append(
            GradientRequestCoreInputs(
                alias=batch.alias,
                edge_index=batch.edge_index.detach().cpu().numpy(),
                coordinates_um=coordinates,
                gene_names=genes,
                fixed_mask=fixed_mask.mask,
                fixed_mask_seed=fixed_mask.seed,
                fixed_mask_sha256=fixed_mask.checksum_sha256,
                graph_sha256=graph_sha256,
            )
        )
    return tuple(inputs)


def _validated_core(core: GradientRequestCoreInputs) -> GradientRequestCoreInputs:
    alias = str(core.alias).strip().upper()
    edges = np.asarray(core.edge_index)
    coordinates = np.asarray(core.coordinates_um, dtype=np.float64)
    mask = np.asarray(core.fixed_mask)
    genes = tuple(str(name) for name in core.gene_names)
    if alias not in CANCER_ALIASES:
        raise LockedGradientRequestError(f"Unknown Cancer core alias {alias!r}.")
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise LockedGradientRequestError("edge_index must be integral [2, edges].")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise LockedGradientRequestError("coordinates_um must have shape [nodes, 2].")
    if not np.isfinite(coordinates).all():
        raise LockedGradientRequestError("coordinates_um must be finite.")
    if mask.dtype != np.bool_ or mask.shape != (len(coordinates), len(genes)):
        raise LockedGradientRequestError(
            "fixed_mask must be boolean [nodes, ordered genes]."
        )
    if not genes or any(not name for name in genes) or len(set(genes)) != len(genes):
        raise LockedGradientRequestError("gene_names must be non-empty and unique.")
    if not _is_sha256(core.fixed_mask_sha256) or not _is_sha256(core.graph_sha256):
        raise LockedGradientRequestError("Graph and fixed-mask checksums must be SHA-256.")
    if isinstance(core.fixed_mask_seed, bool) or int(core.fixed_mask_seed) < 0:
        raise LockedGradientRequestError("fixed_mask_seed must be non-negative.")
    if edges.shape[1] == 0:
        raise LockedGradientRequestError(f"Prepared graph is empty for {alias}.")

    previous_code = -1
    n_nodes = len(coordinates)
    for start in range(0, edges.shape[1], _EDGE_SCAN_CHUNK_SIZE):
        stop = min(start + _EDGE_SCAN_CHUNK_SIZE, edges.shape[1])
        source = edges[0, start:stop].astype(np.int64, copy=False)
        receiver = edges[1, start:stop].astype(np.int64, copy=False)
        if (
            np.any(source < 0)
            or np.any(receiver < 0)
            or np.any(source >= n_nodes)
            or np.any(receiver >= n_nodes)
            or np.any(source == receiver)
        ):
            raise LockedGradientRequestError(
                f"Prepared graph has an invalid edge for {alias}."
            )
        codes = receiver * n_nodes + source
        if int(codes[0]) <= previous_code or np.any(codes[1:] <= codes[:-1]):
            raise LockedGradientRequestError(
                "Prepared graph must be unique receiver-major/source-major sorted."
            )
        previous_code = int(codes[-1])
    return GradientRequestCoreInputs(
        alias=alias,
        edge_index=edges,
        coordinates_um=coordinates,
        gene_names=genes,
        fixed_mask=mask,
        fixed_mask_seed=int(core.fixed_mask_seed),
        fixed_mask_sha256=str(core.fixed_mask_sha256),
        graph_sha256=str(core.graph_sha256),
    )


def _shell_indices(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    bounds = np.asarray([shell[1] for shell in RADIAL_SHELLS], dtype=np.float64)
    result = np.searchsorted(bounds, values, side="left")
    if (
        not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or np.any(result >= len(RADIAL_SHELLS))
    ):
        raise LockedGradientRequestError(
            "Prepared graph contains a non-finite, zero-length, or >500 um edge."
        )
    return result.astype(np.int8, copy=False)


def _selected_shell_edges(
    core: GradientRequestCoreInputs,
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(edge_id, rank, count)`` once per shell without large copies."""

    edges = core.edge_index
    coordinates = core.coordinates_um
    counts = np.zeros(len(RADIAL_SHELLS), dtype=np.int64)
    for start in range(0, edges.shape[1], _EDGE_SCAN_CHUNK_SIZE):
        stop = min(start + _EDGE_SCAN_CHUNK_SIZE, edges.shape[1])
        source = edges[0, start:stop].astype(np.int64, copy=False)
        receiver = edges[1, start:stop].astype(np.int64, copy=False)
        distances = np.linalg.norm(
            coordinates[source] - coordinates[receiver], axis=1
        )
        shell_indices = _shell_indices(distances)
        counts += np.bincount(
            shell_indices, minlength=len(RADIAL_SHELLS)
        ).astype(np.int64, copy=False)
    if np.any(counts <= 0):
        empty = [
            RADIAL_SHELLS[index][2]
            for index in np.flatnonzero(counts <= 0).tolist()
        ]
        raise LockedGradientRequestError(
            f"Prepared graph for {core.alias} lacks shell candidates: {empty}."
        )

    ranks = np.asarray(
        [
            _selection_index(
                role="directed-edge",
                alias=core.alias,
                shell=shell[2],
                count=int(counts[index]),
            )
            for index, shell in enumerate(RADIAL_SHELLS)
        ],
        dtype=np.int64,
    )
    preceding = np.zeros(len(RADIAL_SHELLS), dtype=np.int64)
    selected = np.full(len(RADIAL_SHELLS), -1, dtype=np.int64)
    for start in range(0, edges.shape[1], _EDGE_SCAN_CHUNK_SIZE):
        stop = min(start + _EDGE_SCAN_CHUNK_SIZE, edges.shape[1])
        source = edges[0, start:stop].astype(np.int64, copy=False)
        receiver = edges[1, start:stop].astype(np.int64, copy=False)
        distances = np.linalg.norm(
            coordinates[source] - coordinates[receiver], axis=1
        )
        shell_indices = _shell_indices(distances)
        for shell_index in range(len(RADIAL_SHELLS)):
            positions = np.flatnonzero(shell_indices == shell_index)
            rank = int(ranks[shell_index])
            prior = int(preceding[shell_index])
            if selected[shell_index] < 0 and prior <= rank < prior + len(positions):
                selected[shell_index] = start + int(positions[rank - prior])
            preceding[shell_index] += len(positions)
    if np.any(selected < 0) or not np.array_equal(preceding, counts):
        raise LockedGradientRequestError("Canonical shell selection did not converge.")
    return tuple(
        (int(selected[index]), int(ranks[index]), int(counts[index]))
        for index in range(len(RADIAL_SHELLS))
    )


def generate_locked_gradient_requests(
    cores: Sequence[GradientRequestCoreInputs],
) -> tuple[LockedGradientRequest, ...]:
    """Generate the exact 24 checkpoint-independent requests in locked order."""

    materialized = tuple(_validated_core(core) for core in cores)
    aliases = tuple(core.alias for core in materialized)
    if aliases != tuple(CANCER_ALIASES):
        raise LockedGradientRequestError(
            "Gradient probe inputs must use the exact ordered six Cancer aliases."
        )
    requests: list[LockedGradientRequest] = []
    for core in materialized:
        for shell_index, (edge_id, shell_rank, candidate_count) in enumerate(
            _selected_shell_edges(core)
        ):
            lower, upper, shell_name = RADIAL_SHELLS[shell_index]
            source_node = int(core.edge_index[0, edge_id])
            receiver_node = int(core.edge_index[1, edge_id])
            distance = float(
                np.linalg.norm(
                    core.coordinates_um[source_node]
                    - core.coordinates_um[receiver_node]
                )
            )
            if not lower < distance <= upper:
                raise LockedGradientRequestError(
                    "Selected edge does not belong to its locked radial shell."
                )
            source_feature = _cyclic_feature(
                ~core.fixed_mask[source_node],
                role="observed-source-feature",
                alias=core.alias,
                shell=shell_name,
            )
            target_feature = _cyclic_feature(
                core.fixed_mask[receiver_node],
                role="masked-target-feature",
                alias=core.alias,
                shell=shell_name,
                excluded_index=source_feature,
            )
            if bool(core.fixed_mask[source_node, source_feature]):
                raise LockedGradientRequestError("Selected source feature is masked.")
            if not bool(core.fixed_mask[receiver_node, target_feature]):
                raise LockedGradientRequestError("Selected target feature is observed.")
            if source_feature == target_feature:
                raise LockedGradientRequestError(
                    "Source and target feature indices must be distinct."
                )
            requests.append(
                LockedGradientRequest(
                    request_id=(
                        f"rqkv-grad-{core.alias}-shell-{shell_index:02d}"
                    ),
                    core_alias=core.alias,
                    canonical_edge_id=edge_id,
                    canonical_shell_candidate_index=shell_rank,
                    shell_candidate_count=candidate_count,
                    source_node=source_node,
                    receiver_node=receiver_node,
                    source_feature_index=source_feature,
                    source_feature_name=core.gene_names[source_feature],
                    target_feature_index=target_feature,
                    target_feature_name=core.gene_names[target_feature],
                    distance_um=distance,
                    radial_shell_index=shell_index,
                    radial_shell=shell_name,
                    graph_sha256=core.graph_sha256,
                    fixed_mask_seed=core.fixed_mask_seed,
                    fixed_mask_sha256=core.fixed_mask_sha256,
                )
            )
    if len(requests) != EXPECTED_REQUEST_COUNT or len(
        {request.request_id for request in requests}
    ) != EXPECTED_REQUEST_COUNT:
        raise LockedGradientRequestError(
            "Locked gradient generation must yield exactly 24 unique requests."
        )
    return tuple(requests)


def locked_gradient_request_rows(
    requests: Sequence[LockedGradientRequest],
) -> tuple[dict[str, str], ...]:
    """Return canonical CSV-compatible rows in request order."""

    materialized = tuple(requests)
    if len(materialized) != EXPECTED_REQUEST_COUNT:
        raise LockedGradientRequestError("The locked table must contain 24 requests.")
    return tuple(request.to_csv_row() for request in materialized)


def render_locked_gradient_requests_csv(
    requests: Sequence[LockedGradientRequest],
) -> bytes:
    """Render canonical UTF-8 CSV bytes with stable headers and newlines."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=REQUEST_CSV_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(locked_gradient_request_rows(requests))
    return output.getvalue().encode("utf-8")


def _sidecar_bytes(checksum: str, filename: str) -> bytes:
    return f"{checksum}  {filename}\n".encode("ascii")


def _verified_sidecar(target: Path, sidecar: Path) -> str:
    try:
        raw = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise LockedGradientRequestError(
            f"Cannot read checksum sidecar: {sidecar}."
        ) from exc
    lines = raw.splitlines()
    if len(lines) != 1:
        raise LockedGradientRequestError("Checksum sidecar must contain one line.")
    pieces = lines[0].split("  ", maxsplit=1)
    if len(pieces) != 2 or not _is_sha256(pieces[0]) or pieces[1] != target.name:
        raise LockedGradientRequestError(
            "Checksum sidecar must bind the exact target basename."
        )
    observed = _sha256_file(target)
    if observed != pieces[0]:
        raise LockedGradientRequestError(
            f"Checksum mismatch for locked artifact {target.name}."
        )
    return observed


def freeze_locked_gradient_request_csv(
    requests: Sequence[LockedGradientRequest],
    *,
    request_csv_path: str | Path,
    request_sha256_path: str | Path,
) -> str:
    """Write a new immutable CSV plus sidecar, refusing every overwrite."""

    csv_path = Path(request_csv_path)
    sha_path = Path(request_sha256_path)
    if csv_path.exists() or sha_path.exists():
        raise FileExistsError(
            "Locked request CSV or checksum already exists; refusing overwrite."
        )
    if csv_path.parent != sha_path.parent:
        raise LockedGradientRequestError(
            "Request CSV and checksum sidecar must share one directory."
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_locked_gradient_requests_csv(requests)
    checksum = _sha256_bytes(content)
    temporary = Path(tempfile.mkdtemp(prefix=".gradient-requests-", dir=csv_path.parent))
    try:
        temp_csv = temporary / csv_path.name
        temp_sha = temporary / sha_path.name
        temp_csv.write_bytes(content)
        temp_sha.write_bytes(_sidecar_bytes(checksum, csv_path.name))
        created: list[Path] = []
        try:
            # Hard-link publication is exclusive: unlike os.replace, a racing
            # generator can never overwrite a table that appeared after the
            # initial existence check.  On an ordinary exception the first link
            # is rolled back if the second cannot be published.
            os.link(temp_csv, csv_path)
            created.append(csv_path)
            os.link(temp_sha, sha_path)
            created.append(sha_path)
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        temp_csv.unlink()
        temp_sha.unlink()
    finally:
        (temporary / csv_path.name).unlink(missing_ok=True)
        (temporary / sha_path.name).unlink(missing_ok=True)
        try:
            temporary.rmdir()
        except OSError:
            pass
    return checksum


def verify_locked_gradient_protocol(
    protocol_path: str | Path,
    protocol_sha256_path: str | Path,
) -> Mapping[str, Any]:
    """Checksum and validate the additive four-seed analysis protocol."""

    path = Path(protocol_path)
    _verified_sidecar(path, Path(protocol_sha256_path))
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LockedGradientRequestError("Cannot parse analysis protocol YAML.") from exc
    if not isinstance(value, Mapping):
        raise LockedGradientRequestError("Analysis protocol must be a mapping.")
    required = {
        "analysis_protocol_schema": GRADIENT_PROTOCOL_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise LockedGradientRequestError(
                f"Analysis protocol {field} does not match the locked value."
            )
    ensemble = value.get("ensemble")
    extraction = value.get("fixed_probe_extraction")
    gradient = value.get("selected_gradient_requests")
    if not all(isinstance(section, Mapping) for section in (ensemble, extraction, gradient)):
        raise LockedGradientRequestError("Analysis protocol sections are incomplete.")
    assert isinstance(ensemble, Mapping)
    assert isinstance(extraction, Mapping)
    assert isinstance(gradient, Mapping)
    if tuple(ensemble.get("active_model_seeds", ())) != EXPECTED_ACTIVE_MODEL_SEEDS:
        raise LockedGradientRequestError("Active analysis seeds must be 0 through 3.")
    if tuple(ensemble.get("deferred_model_seeds", ())) != DEFERRED_MODEL_SEEDS:
        raise LockedGradientRequestError("Seed 4 must remain deferred.")
    if (
        extraction.get("layer") != FINAL_LAYER
        or extraction.get("receiver_probes_per_core") != 64
        or extraction.get("top_edge_fraction") != 0.05
        or extraction.get("mutual_top_edges_per_core_per_seed") != 100
        or tuple(extraction.get("empirical_quantiles", ()))
        != (0.05, 0.25, 0.75, 0.95)
    ):
        raise LockedGradientRequestError("Fixed-probe extraction settings drifted.")
    if (
        gradient.get("request_schema") != GRADIENT_REQUEST_SCHEMA
        or gradient.get("request_count") != EXPECTED_REQUEST_COUNT
        or gradient.get("requests_per_core") != len(RADIAL_SHELLS)
        or tuple(gradient.get("radial_shells", ()))
        != tuple(shell[2] for shell in RADIAL_SHELLS)
        or not _is_sha256(gradient.get("request_table_sha256"))
    ):
        raise LockedGradientRequestError("Selected-gradient protocol settings drifted.")
    return value


def load_and_verify_locked_gradient_requests(
    request_csv_path: str | Path,
    request_sha256_path: str | Path,
    *,
    cores: Sequence[GradientRequestCoreInputs],
    protocol: Mapping[str, Any] | None = None,
) -> tuple[LockedGradientRequest, ...]:
    """Gate a request table by checksum and exact canonical regeneration."""

    csv_path = Path(request_csv_path)
    observed_sha256 = _verified_sidecar(csv_path, Path(request_sha256_path))
    if protocol is not None:
        section = protocol.get("selected_gradient_requests")
        if not isinstance(section, Mapping):
            raise LockedGradientRequestError(
                "Analysis protocol lacks selected_gradient_requests."
            )
        if section.get("request_table_sha256") != observed_sha256:
            raise LockedGradientRequestError(
                "Request CSV does not match the checksum locked in the protocol."
            )
    expected = generate_locked_gradient_requests(cores)
    canonical = render_locked_gradient_requests_csv(expected)
    try:
        observed = csv_path.read_bytes()
    except OSError as exc:
        raise LockedGradientRequestError("Cannot read locked request CSV.") from exc
    if observed != canonical:
        raise LockedGradientRequestError(
            "Locked request CSV differs from canonical prepared-input regeneration."
        )
    return expected


def group_selected_derivative_requests(
    requests: Sequence[LockedGradientRequest],
) -> dict[str, tuple[Any, ...]]:
    """Convert the exact table to existing derivative requests grouped by core."""

    from .relative_qkv_post_training import SelectedDerivativeRequest

    materialized = tuple(requests)
    if len(materialized) != EXPECTED_REQUEST_COUNT:
        raise LockedGradientRequestError("Exactly 24 requests are required.")
    groups: dict[str, list[Any]] = {alias: [] for alias in CANCER_ALIASES}
    expected_order = [
        (alias, shell_index)
        for alias in CANCER_ALIASES
        for shell_index in range(len(RADIAL_SHELLS))
    ]
    observed_order = [
        (request.core_alias, int(request.radial_shell_index))
        for request in materialized
    ]
    if observed_order != expected_order:
        raise LockedGradientRequestError(
            "Derivative requests are not in canonical core/shell order."
        )
    for request in materialized:
        groups[request.core_alias].append(
            SelectedDerivativeRequest(
                request_id=request.request_id,
                core_alias=request.core_alias,
                source_node=request.source_node,
                source_feature=request.source_feature_index,
                receiver_node=request.receiver_node,
                target_feature=request.target_feature_index,
                attention_head=None,
                layer=FINAL_LAYER,
            )
        )
    return {alias: tuple(groups[alias]) for alias in CANCER_ALIASES}


__all__ = [
    "ATTENTION_HEAD",
    "CAMPAIGN_ID",
    "DEFERRED_MODEL_SEEDS",
    "EXPECTED_ACTIVE_MODEL_SEEDS",
    "EXPECTED_REQUEST_COUNT",
    "FINAL_LAYER",
    "GRADIENT_PROTOCOL_SCHEMA",
    "GRADIENT_REQUEST_SCHEMA",
    "GradientRequestCoreInputs",
    "LockedGradientRequest",
    "LockedGradientRequestError",
    "RADIAL_SHELLS",
    "freeze_locked_gradient_request_csv",
    "generate_locked_gradient_requests",
    "group_selected_derivative_requests",
    "load_and_verify_locked_gradient_requests",
    "load_prepared_gradient_request_inputs",
    "locked_gradient_request_rows",
    "render_locked_gradient_requests_csv",
    "verify_locked_gradient_protocol",
]

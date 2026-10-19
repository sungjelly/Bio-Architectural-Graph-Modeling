"""Joint six-core clustering of intrinsic and contextual relative-QKV embeddings.

This is a locked, post-training readout.  ``h0`` is the output of the trained
``NodeEncoder`` and ``hL`` is the output of the final graph block immediately
before the expression decoder.  Their difference and norm describe model
representation change only; they are not biological influence or causality.

The workflow is deliberately stage-resumable.  A checksum-valid extraction
receipt skips model inference, and a checksum-valid clustering receipt skips
PCA, kNN construction, and Leiden when only plotting remains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import hmac
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .cancer_pooled_full_core import (
    CANCER_ALIASES,
    CORE_ALIAS_BY_NUMBER,
    CORE_NUMBERS,
)
from .checkpoint_catalog import resolve_checkpoint
from .fingerprints import sha256_file
from .paths import ProjectPaths
from .registry import Registry
from .relative_qkv_post_training import (
    CAMPAIGN_ID,
    CHECKPOINT_SCHEMA,
    load_core_coordinates_and_genes,
    load_prepared_relative_qkv_batches,
    load_relative_qkv_checkpoint,
)
from .run_archive import RunArchive, verify_run_bundle


ANALYSIS_SCHEMA = "cancer_6core_embedding_clustering_v1"
EXTRACTION_SCHEMA = "cancer_6core_embedding_extraction_v1"
CLUSTERING_SCHEMA = "cancer_6core_joint_embedding_clustering_v1"
EXPECTED_RUN_ID = "r_20260824T121803Z_16144620_s000_f00_a02_62498796"
EXPECTED_MODEL_SEED = 0
EXPECTED_CELL_COUNTS = {
    1: 8_924,
    9: 17_223,
    13: 5_345,
    15: 38_145,
    21: 4_897,
    23: 43_462,
}
EXPECTED_TOTAL_CELLS = sum(EXPECTED_CELL_COUNTS.values())
REPRESENTATIONS = ("intrinsic", "contextual")
REPRESENTATION_ARRAY = {"intrinsic": "h0", "contextual": "hL"}
REPRESENTATION_PREFIX = {"intrinsic": "I", "contextual": "C"}
DEFAULT_N_NEIGHBORS = 30
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_RANDOM_SEED = 20260825
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
PREDICTION_INVARIANCE_ATOL = 1.0e-5
PREDICTION_INVARIANCE_RTOL = 1.0e-6


class EmbeddingClusterAnalysisError(ValueError):
    """Raised when the locked extraction or analysis contract is violated."""


@dataclass(frozen=True, slots=True)
class ResolvedAnalysisInputs:
    run_id: str
    project_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_payload: Mapping[str, Any] = field(repr=False)
    bundle_path: Path
    cohort_dir: Path
    graph_dir: Path
    cohort_manifest: Mapping[str, Any] = field(repr=False)
    graph_manifest: Mapping[str, Any] = field(repr=False)
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CoreEmbeddings:
    alias: str
    core_number: int
    cell_index: np.ndarray = field(repr=False)
    coordinates_um: np.ndarray = field(repr=False)
    h0: np.ndarray = field(repr=False)
    hL: np.ndarray = field(repr=False)
    delta_h: np.ndarray = field(repr=False)
    delta_h_l2: np.ndarray = field(repr=False)

    @property
    def n_cells(self) -> int:
        return int(len(self.cell_index))


@dataclass(frozen=True, slots=True)
class PCAResult:
    normalized_scores: np.ndarray = field(repr=False)
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class KNNGraphResult:
    edge_pairs: np.ndarray = field(repr=False)
    receipt: Mapping[str, Any]
    # Optional directed rows support bounded exact-recall audits without
    # rebuilding an approximate index.  Existing consumers use only the
    # undirected edge union and remain backward compatible.
    directed_neighbors: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class LeidenResult:
    labels: np.ndarray = field(repr=False)
    receipt: Mapping[str, Any]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingClusterAnalysisError("JSON output cannot contain NaN or Inf.")
        return number
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EmbeddingClusterAnalysisError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EmbeddingClusterAnalysisError(f"{label} must contain a JSON object.")
    return value


def _read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise EmbeddingClusterAnalysisError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EmbeddingClusterAnalysisError(f"{label} must contain a mapping.")
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write compressed NPZ arrays with sorted names and fixed ZIP metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for name in sorted(arrays):
                if not name or "/" in name or "\\" in name:
                    raise EmbeddingClusterAnalysisError(
                        f"Invalid NPZ array name: {name!r}."
                    )
                array = np.ascontiguousarray(np.asarray(arrays[name]))
                if array.dtype.hasobject:
                    raise EmbeddingClusterAnalysisError(
                        "Embedding NPZ arrays may not use object dtype."
                    )
                stream = BytesIO()
                np.lib.format.write_array(stream, array, allow_pickle=False)
                info = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    stream.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.lib.format.write_array(
                handle,
                np.ascontiguousarray(array),
                allow_pickle=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise EmbeddingClusterAnalysisError(
            "PyArrow is required for the cell-level Parquet output."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            version="2.6",
            write_statistics=True,
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _tensor_sha256(name: str, value: torch.Tensor) -> str:
    return _array_sha256(name, value.detach().cpu().contiguous().numpy())


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        if path.is_symlink():
            raise EmbeddingClusterAnalysisError(
                "Analysis outputs may not contain symbolic links."
            )
        files[relative] = _file_record(path)
    return files


def _receipt_with_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    receipt["manifest_content_sha256"] = _canonical_sha256(receipt)
    return receipt


def _verify_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    content = dict(value)
    observed = str(content.pop("manifest_content_sha256", ""))
    expected = _canonical_sha256(content)
    if not hmac.compare_digest(observed, expected):
        raise EmbeddingClusterAnalysisError(f"{label} self-checksum is invalid.")


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _git_provenance(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments], stderr=subprocess.DEVNULL
        )

    try:
        commit = run("rev-parse", "HEAD").decode("ascii").strip()
        status = run("status", "--porcelain=v1")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _project_artifact_path(raw: object, *, paths: ProjectPaths) -> Path:
    value = Path(str(raw))
    if value.is_absolute():
        return value.resolve(strict=True)
    parts = value.parts
    if parts and parts[0] == "data":
        return paths.data_root.joinpath(*parts[1:]).resolve(strict=True)
    return (paths.project_root / value).resolve(strict=True)


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EmbeddingClusterAnalysisError(
            f"Cannot load checkpoint identity: {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise EmbeddingClusterAnalysisError("Checkpoint payload must be a mapping.")
    return dict(value)


def resolve_analysis_inputs(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None,
    checkpoint: str | Path | None,
) -> ResolvedAnalysisInputs:
    """Resolve and verify the immutable checkpoint and prepared input bundle."""

    if run_id is None and checkpoint is None:
        raise EmbeddingClusterAnalysisError(
            "Supply --run-id or an explicit --checkpoint."
        )
    if checkpoint is None:
        assert run_id is not None
        canonical_run_id = registry.resolve_run_id(str(run_id))
        checkpoint_path = resolve_checkpoint(
            registry, canonical_run_id, paths, role="last"
        )
        canonical_final_checkpoint = checkpoint_path
        payload = _load_checkpoint_payload(checkpoint_path)
    else:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = paths.project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve(strict=True)
        payload = _load_checkpoint_payload(checkpoint_path)
        payload_run_id = str(payload.get("run_id", ""))
        if not payload_run_id:
            raise EmbeddingClusterAnalysisError("Checkpoint does not declare run_id.")
        canonical_run_id = registry.resolve_run_id(payload_run_id)
        if run_id is not None and registry.resolve_run_id(str(run_id)) != canonical_run_id:
            raise EmbeddingClusterAnalysisError(
                "--run-id disagrees with the explicit checkpoint payload."
            )
        canonical_final_checkpoint = resolve_checkpoint(
            registry, canonical_run_id, paths, role="last"
        )
        if not hmac.compare_digest(
            sha256_file(checkpoint_path), sha256_file(canonical_final_checkpoint)
        ):
            raise EmbeddingClusterAnalysisError(
                "Explicit checkpoint is not content-identical to the verified "
                "canonical final checkpoint for its run."
            )

    if payload.get("run_id") != canonical_run_id:
        raise EmbeddingClusterAnalysisError("Resolved checkpoint run identity drifted.")
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise EmbeddingClusterAnalysisError("Unsupported relative-QKV checkpoint schema.")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise EmbeddingClusterAnalysisError("Checkpoint belongs to another campaign.")
    if (
        canonical_run_id != EXPECTED_RUN_ID
        or int(payload.get("model_seed", -1)) != EXPECTED_MODEL_SEED
    ):
        raise EmbeddingClusterAnalysisError(
            "This analysis contract is locked to the selected completed seed-0 run."
        )
    run_record = registry.show_run(canonical_run_id)
    if run_record is None or str(run_record.get("status")) != "completed":
        raise EmbeddingClusterAnalysisError("Selected run is not completed in the registry.")

    bundle_path = RunArchive.artifact_path_for(canonical_run_id, paths)
    verify_run_bundle(bundle_path)
    resolved_config = _read_yaml(
        bundle_path / "config.resolved.yaml", label="resolved run configuration"
    )
    dataset = resolved_config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise EmbeddingClusterAnalysisError("Resolved run lacks dataset configuration.")
    cohort_dir = _project_artifact_path(dataset.get("prepared_artifact"), paths=paths)
    graph_dir = _project_artifact_path(
        dataset.get("prepared_graph_artifact"), paths=paths
    )
    cohort_manifest_path = cohort_dir / "manifest.json"
    graph_manifest_path = graph_dir / "manifest.json"
    cohort_sha = sha256_file(cohort_manifest_path)
    graph_sha = sha256_file(graph_manifest_path)
    if cohort_sha != dataset.get("cohort_manifest_file_sha256"):
        raise EmbeddingClusterAnalysisError("Cohort manifest checksum changed from run config.")
    if graph_sha != dataset.get("graph_manifest_file_sha256"):
        raise EmbeddingClusterAnalysisError("Graph manifest checksum changed from run config.")
    cohort_manifest = _read_json(cohort_manifest_path, label="cohort manifest")
    graph_manifest = _read_json(graph_manifest_path, label="graph manifest")
    if tuple(cohort_manifest.get("cohort", {}).get("aliases", ())) != CANCER_ALIASES:
        raise EmbeddingClusterAnalysisError("Cohort aliases do not match locked order.")
    if tuple(graph_manifest.get("aliases", ())) != CANCER_ALIASES:
        raise EmbeddingClusterAnalysisError("Graph aliases do not match locked order.")

    training_git = _read_json(bundle_path / "provenance/git.json", label="run Git provenance")
    construction = payload.get("model_construction")
    if not isinstance(construction, Mapping):
        raise EmbeddingClusterAnalysisError("Checkpoint lacks model construction receipt.")
    provenance: dict[str, Any] = {
        "selection_policy": (
            "completed_seed_0_fallback_after_no_analysis_or_campaign_primary"
            if canonical_run_id == EXPECTED_RUN_ID
            else "explicit_run_or_checkpoint"
        ),
        "campaign_id": CAMPAIGN_ID,
        "run_id": canonical_run_id,
        "model_seed": int(payload.get("model_seed", -1)),
        "checkpoint": {
            "relative_path": checkpoint_path.relative_to(paths.project_root).as_posix()
            if checkpoint_path.is_relative_to(paths.project_root)
            else checkpoint_path.as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "canonical_final_path": (
                canonical_final_checkpoint.relative_to(paths.project_root).as_posix()
                if canonical_final_checkpoint.is_relative_to(paths.project_root)
                else canonical_final_checkpoint.as_posix()
            ),
            "schema": payload.get("checkpoint_schema"),
            "completed_global_epochs": int(payload.get("completed_global_epochs", -1)),
            "model_state_sha256": payload.get("model_state_checksum"),
        },
        "training_source": training_git,
        "analysis_source": _git_provenance(paths.project_root),
        "model_construction": dict(construction),
        "dataset_fingerprint": dataset.get("dataset_fingerprint"),
        "preprocessing_version": dataset.get("preprocessing_version"),
        "cohort_manifest": {
            "relative_path": cohort_manifest_path.relative_to(paths.project_root).as_posix()
            if cohort_manifest_path.is_relative_to(paths.project_root)
            else cohort_manifest_path.as_posix(),
            "file_sha256": cohort_sha,
            "content_sha256": cohort_manifest.get("manifest_content_sha256"),
            "preprocessing_statistics_sha256": cohort_manifest.get("files", {}).get(
                "cohort_statistics.npz"
            ),
            "statistics_component_checksums": cohort_manifest.get("preprocessing", {}).get(
                "statistics_checksums", {}
            ),
        },
        "graph_manifest": {
            "relative_path": graph_manifest_path.relative_to(paths.project_root).as_posix()
            if graph_manifest_path.is_relative_to(paths.project_root)
            else graph_manifest_path.as_posix(),
            "file_sha256": graph_sha,
            "content_sha256": graph_manifest.get("manifest_content_sha256"),
        },
    }
    checkpoint_sha256 = str(provenance["checkpoint"]["sha256"])
    resolved = ResolvedAnalysisInputs(
        run_id=canonical_run_id,
        project_root=paths.project_root,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_payload=payload,
        bundle_path=bundle_path,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
        cohort_manifest=cohort_manifest,
        graph_manifest=graph_manifest,
        provenance=provenance,
    )
    _verify_prepared_input_files(resolved)
    return resolved


def _verify_prepared_input_files(inputs: ResolvedAnalysisInputs) -> None:
    """Hash every prepared cohort, graph, and geometry file fail-closed."""

    cohort_files = inputs.cohort_manifest.get("files")
    graph_cores = inputs.graph_manifest.get("cores")
    if not isinstance(cohort_files, Mapping) or not isinstance(graph_cores, list):
        raise EmbeddingClusterAnalysisError("Prepared input manifests are incomplete.")
    checks: list[tuple[Path, str, str]] = []
    for relative, expected in sorted(cohort_files.items()):
        checks.append(
            (
                inputs.cohort_dir / str(relative),
                str(expected),
                f"cohort:{relative}",
            )
        )
    for record in graph_cores:
        if not isinstance(record, Mapping) or not isinstance(record.get("files"), Mapping):
            raise EmbeddingClusterAnalysisError("Graph core file receipt is malformed.")
        alias = str(record.get("alias", ""))
        for filename, expected in sorted(record["files"].items()):
            checks.append(
                (
                    inputs.graph_dir / "cores" / alias / str(filename),
                    str(expected),
                    f"graph:{alias}/{filename}",
                )
            )
    for path, expected, label in checks:
        if not path.is_file():
            raise EmbeddingClusterAnalysisError(
                f"Prepared input file is missing: {label}."
            )
        observed = sha256_file(path)
        if not hmac.compare_digest(observed, expected):
            raise EmbeddingClusterAnalysisError(
                f"Prepared input checksum changed: {label}."
            )


def _expected_core_records(inputs: ResolvedAnalysisInputs) -> tuple[dict[str, Any], ...]:
    cohort_records = inputs.cohort_manifest.get("cores")
    graph_records = inputs.graph_manifest.get("cores")
    if not isinstance(cohort_records, list) or not isinstance(graph_records, list):
        raise EmbeddingClusterAnalysisError("Input manifests lack core records.")
    cohort_by_alias = {
        str(record.get("alias")): record
        for record in cohort_records
        if isinstance(record, Mapping)
    }
    graph_by_alias = {
        str(record.get("alias")): record
        for record in graph_records
        if isinstance(record, Mapping)
    }
    files = inputs.cohort_manifest.get("files")
    if not isinstance(files, Mapping):
        raise EmbeddingClusterAnalysisError("Cohort manifest lacks file checksums.")
    result: list[dict[str, Any]] = []
    for core_number, alias in zip(CORE_NUMBERS, CANCER_ALIASES, strict=True):
        cohort_record = cohort_by_alias.get(alias)
        graph_record = graph_by_alias.get(alias)
        if not isinstance(cohort_record, Mapping) or not isinstance(graph_record, Mapping):
            raise EmbeddingClusterAnalysisError(f"Missing input record for {alias}.")
        relative = f"cores/{alias}.npz"
        result.append(
            {
                "alias": alias,
                "core_number": int(core_number),
                "cell_count": int(cohort_record["cell_count"]),
                "prepared_core_artifact_sha256": str(files[relative]),
                "prepared_component_checksums": dict(
                    cohort_record.get("component_checksums", {})
                ),
                "graph_record_sha256": str(graph_record.get("record_sha256")),
                "graph_logical_sha256": str(
                    graph_record.get("graph", {}).get("checksums", {}).get(
                        "graph_sha256"
                    )
                ),
                "graph_file_checksums": dict(graph_record.get("files", {})),
            }
        )
    if sum(record["cell_count"] for record in result) != int(
        inputs.cohort_manifest.get("cohort", {}).get("total_cells", -1)
    ):
        raise EmbeddingClusterAnalysisError("Core counts do not sum to cohort total.")
    if {
        int(record["core_number"]): int(record["cell_count"])
        for record in result
    } != EXPECTED_CELL_COUNTS:
        raise EmbeddingClusterAnalysisError(
            "Prepared core cell counts changed from the locked analysis contract."
        )
    return tuple(result)


def _core_embedding_path(output_root: Path, core_number: int) -> Path:
    return output_root / "embeddings" / f"core_{int(core_number)}_embeddings.npz"


def _validate_core_embedding_arrays(
    *, alias: str, core_number: int, arrays: Mapping[str, np.ndarray], expected_cells: int
) -> CoreEmbeddings:
    required = {
        "cell_index",
        "core_number",
        "coordinates_um",
        "h0",
        "hL",
        "delta_h",
        "delta_h_l2",
    }
    if set(arrays) != required:
        raise EmbeddingClusterAnalysisError(
            f"Embedding arrays for {alias} have an unexpected schema."
        )
    cell_index = np.asarray(arrays["cell_index"], dtype=np.int64)
    core_number_array = np.asarray(arrays["core_number"])
    coordinates = np.asarray(arrays["coordinates_um"], dtype=np.float64)
    h0 = np.asarray(arrays["h0"], dtype=np.float32)
    hL = np.asarray(arrays["hL"], dtype=np.float32)
    delta = np.asarray(arrays["delta_h"], dtype=np.float32)
    norms = np.asarray(arrays["delta_h_l2"], dtype=np.float64)
    if not np.array_equal(cell_index, np.arange(expected_cells, dtype=np.int64)):
        raise EmbeddingClusterAnalysisError(f"Cell order changed for {alias}.")
    if (
        core_number_array.size != 1
        or int(core_number_array.reshape(-1)[0]) != int(core_number)
    ):
        raise EmbeddingClusterAnalysisError(f"Core number changed for {alias}.")
    if coordinates.shape != (expected_cells, 2):
        raise EmbeddingClusterAnalysisError(f"Coordinate shape changed for {alias}.")
    if h0.ndim != 2 or h0.shape[0] != expected_cells or hL.shape != h0.shape:
        raise EmbeddingClusterAnalysisError(f"h0/hL shapes are invalid for {alias}.")
    if delta.shape != h0.shape or norms.shape != (expected_cells,):
        raise EmbeddingClusterAnalysisError(f"Delta arrays are invalid for {alias}.")
    if not all(
        np.isfinite(value).all() for value in (coordinates, h0, hL, delta, norms)
    ):
        raise EmbeddingClusterAnalysisError(f"Non-finite extraction values for {alias}.")
    if np.any(norms < 0.0):
        raise EmbeddingClusterAnalysisError(f"Negative delta norm for {alias}.")
    expected_delta = hL - h0
    if not np.array_equal(delta, expected_delta):
        raise EmbeddingClusterAnalysisError(f"Stored delta_h changed for {alias}.")
    expected_norm = np.linalg.norm(delta.astype(np.float64), axis=1)
    if not np.array_equal(norms, expected_norm):
        raise EmbeddingClusterAnalysisError(f"Stored delta_h_l2 changed for {alias}.")
    return CoreEmbeddings(
        alias=alias,
        core_number=int(core_number),
        cell_index=np.ascontiguousarray(cell_index),
        coordinates_um=np.ascontiguousarray(coordinates),
        h0=np.ascontiguousarray(h0),
        hL=np.ascontiguousarray(hL),
        delta_h=np.ascontiguousarray(delta),
        delta_h_l2=np.ascontiguousarray(norms),
    )


def load_core_embedding_file(
    path: Path, *, alias: str, core_number: int, expected_cells: int
) -> CoreEmbeddings:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise EmbeddingClusterAnalysisError(f"Cannot load embedding file: {path}") from exc
    return _validate_core_embedding_arrays(
        alias=alias,
        core_number=core_number,
        arrays=arrays,
        expected_cells=expected_cells,
    )


def _verify_stage_files(
    output_root: Path, expected: Mapping[str, Any], *, label: str
) -> None:
    for relative, record in expected.items():
        if not isinstance(record, Mapping):
            raise EmbeddingClusterAnalysisError(f"{label} file record is malformed.")
        path = output_root / relative
        if not path.is_file() or _file_record(path) != dict(record):
            raise EmbeddingClusterAnalysisError(
                f"{label} output checksum changed: {relative}"
            )


def _verify_extraction_receipt(
    output_root: Path,
    receipt: Mapping[str, Any],
    *,
    inputs: ResolvedAnalysisInputs,
) -> None:
    _verify_self_hash(receipt, label="extraction receipt")
    if (
        receipt.get("schema") != EXTRACTION_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("run_id") != inputs.run_id
        or receipt.get("checkpoint_sha256") != inputs.checkpoint_sha256
        or tuple(receipt.get("core_order", ())) != CORE_NUMBERS
    ):
        raise EmbeddingClusterAnalysisError("Extraction receipt identity is invalid.")
    files = receipt.get("files")
    if not isinstance(files, Mapping) or len(files) != len(CORE_NUMBERS):
        raise EmbeddingClusterAnalysisError("Extraction receipt files are incomplete.")
    expected_cores = _expected_core_records(inputs)
    observed_cores = receipt.get("cores")
    if not isinstance(observed_cores, list) or len(observed_cores) != len(expected_cores):
        raise EmbeddingClusterAnalysisError("Extraction core receipts are incomplete.")
    for observed, expected in zip(observed_cores, expected_cores, strict=True):
        if not isinstance(observed, Mapping) or any(
            observed.get(name) != value for name, value in expected.items()
        ):
            raise EmbeddingClusterAnalysisError(
                "Extraction source-core identity changed from prepared inputs."
            )
    expected_dimension = int(inputs.provenance["model_construction"]["hidden_dim"])
    if (
        int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS
        or int(receipt.get("embedding_dimension", -1)) != expected_dimension
    ):
        raise EmbeddingClusterAnalysisError("Extraction coverage or width changed.")
    observed_provenance = receipt.get("input_provenance")
    if not isinstance(observed_provenance, Mapping):
        raise EmbeddingClusterAnalysisError("Extraction input provenance is missing.")
    stable_provenance_keys = (
        "campaign_id",
        "run_id",
        "model_seed",
        "checkpoint",
        "training_source",
        "model_construction",
        "dataset_fingerprint",
        "preprocessing_version",
        "cohort_manifest",
        "graph_manifest",
    )
    if any(
        observed_provenance.get(name) != inputs.provenance.get(name)
        for name in stable_provenance_keys
    ):
        raise EmbeddingClusterAnalysisError("Extraction input provenance changed.")
    _verify_stage_files(output_root, files, label="extraction")


def _staged_core_tensors(
    batch: Any, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expression = batch.target_expression.to(device=device, dtype=torch.float32)
    gene_mask = torch.zeros_like(expression, dtype=torch.bool, device=device)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    relative_geometry = batch.relative_geometry.to(device=device, dtype=torch.float32)
    if int(torch.count_nonzero(gene_mask).item()) != 0:
        raise EmbeddingClusterAnalysisError("Extraction gene mask is not all-zero.")
    return expression, gene_mask, covariates, edge_index, relative_geometry


def extract_intermediate_embeddings(
    *,
    inputs: ResolvedAnalysisInputs,
    output_root: Path,
    device: str | torch.device,
) -> Mapping[str, Any]:
    """Extract all six complete cores, or reuse a verified extraction stage."""

    receipt_path = output_root / "embeddings" / "extraction_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="extraction receipt")
        _verify_extraction_receipt(output_root, receipt, inputs=inputs)
        return receipt
    embedding_dir = output_root / "embeddings"
    if embedding_dir.exists() and any(embedding_dir.iterdir()):
        raise EmbeddingClusterAnalysisError(
            "Partial embedding outputs exist without a complete receipt; refusing "
            "to overwrite them."
        )
    embedding_dir.mkdir(parents=True, exist_ok=True)

    batches = load_prepared_relative_qkv_batches(
        cohort_dir=inputs.cohort_dir,
        graph_dir=inputs.graph_dir,
    )
    expected_records = _expected_core_records(inputs)
    if tuple(batch.alias for batch in batches) != CANCER_ALIASES:
        raise EmbeddingClusterAnalysisError("Loaded core order changed.")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise EmbeddingClusterAnalysisError("CUDA extraction requested but unavailable.")
    loaded = load_relative_qkv_checkpoint(
        inputs.checkpoint_path,
        num_genes=batches[0].n_genes,
        node_covariate_dim=int(batches[0].node_covariates.shape[1]),
        device=resolved_device,
    )
    model = loaded.model
    model.eval()
    if model.training or any(module.training for module in model.modules()):
        raise EmbeddingClusterAnalysisError("model.eval() did not disable all dropout.")
    attention_dropout = {
        float(block.attention_dropout_probability) for block in model.blocks
    }
    if attention_dropout != {0.0}:
        raise EmbeddingClusterAnalysisError("Checkpoint attention dropout is not zero.")

    core_receipts: list[dict[str, Any]] = []
    file_records: dict[str, dict[str, Any]] = {}
    for batch, expected in zip(batches, expected_records, strict=True):
        alias = str(expected["alias"])
        core_number = int(expected["core_number"])
        expected_cells = int(expected["cell_count"])
        if batch.alias != alias or batch.n_nodes != expected_cells:
            raise EmbeddingClusterAnalysisError(f"Loaded cell count changed for {alias}.")
        coordinates, _ = load_core_coordinates_and_genes(
            inputs.cohort_dir, alias=alias
        )
        if coordinates.shape != (expected_cells, 2):
            raise EmbeddingClusterAnalysisError(
                f"Coordinates do not align with model rows for {alias}."
            )
        expression_before = _tensor_sha256("target_expression", batch.target_expression)
        metadata_before = _tensor_sha256("node_covariates", batch.node_covariates)
        (
            expression,
            gene_mask,
            covariates,
            edge_index,
            relative_geometry,
        ) = _staged_core_tensors(batch, device=resolved_device)
        with torch.inference_mode():
            ordinary = model(
                input_expression=expression,
                gene_mask=gene_mask,
                edge_index=edge_index,
                relative_geometry=relative_geometry,
                node_covariates=covariates,
            )
            extended = model(
                input_expression=expression,
                gene_mask=gene_mask,
                edge_index=edge_index,
                relative_geometry=relative_geometry,
                node_covariates=covariates,
                return_intermediate_embeddings=True,
            )
        prediction_difference = torch.abs(
            ordinary.prediction - extended.prediction
        )
        prediction_exact = bool(torch.equal(ordinary.prediction, extended.prediction))
        prediction_maximum = float(torch.max(prediction_difference).item())
        if not torch.allclose(
            ordinary.prediction,
            extended.prediction,
            rtol=PREDICTION_INVARIANCE_RTOL,
            atol=PREDICTION_INVARIANCE_ATOL,
            equal_nan=False,
        ):
            raise EmbeddingClusterAnalysisError(
                "Intermediate extraction changed predictions for "
                f"{alias} (maximum absolute difference {prediction_maximum}; "
                f"rtol={PREDICTION_INVARIANCE_RTOL}, "
                f"atol={PREDICTION_INVARIANCE_ATOL})."
            )
        if (
            extended.node_encoder_embedding is None
            or extended.final_graph_embedding is None
            or not torch.equal(
                extended.final_graph_embedding, extended.node_embedding
            )
        ):
            raise EmbeddingClusterAnalysisError(
                f"Intermediate model outputs are incomplete for {alias}."
            )
        h0 = np.ascontiguousarray(
            extended.node_encoder_embedding.detach().cpu().float().numpy(),
            dtype=np.float32,
        )
        hL = np.ascontiguousarray(
            extended.final_graph_embedding.detach().cpu().float().numpy(),
            dtype=np.float32,
        )
        delta = np.ascontiguousarray(hL - h0, dtype=np.float32)
        delta_norm = np.ascontiguousarray(
            np.linalg.norm(delta.astype(np.float64), axis=1), dtype=np.float64
        )
        cell_index = np.arange(expected_cells, dtype=np.int64)
        core = _validate_core_embedding_arrays(
            alias=alias,
            core_number=core_number,
            arrays={
                "cell_index": cell_index,
                "core_number": np.asarray(core_number, dtype=np.int16),
                "coordinates_um": coordinates,
                "h0": h0,
                "hL": hL,
                "delta_h": delta,
                "delta_h_l2": delta_norm,
            },
            expected_cells=expected_cells,
        )
        metadata_after = _tensor_sha256("node_covariates", batch.node_covariates)
        expression_after = _tensor_sha256("target_expression", batch.target_expression)
        if metadata_after != metadata_before or expression_after != expression_before:
            raise EmbeddingClusterAnalysisError(
                f"Prepared expression or metadata changed during extraction for {alias}."
            )
        output_path = _core_embedding_path(output_root, core_number)
        _write_deterministic_npz(
            output_path,
            {
                "cell_index": core.cell_index,
                "core_number": np.asarray(core_number, dtype=np.int16),
                "coordinates_um": core.coordinates_um,
                "h0": core.h0,
                "hL": core.hL,
                "delta_h": core.delta_h,
                "delta_h_l2": core.delta_h_l2,
            },
        )
        loaded_core = load_core_embedding_file(
            output_path,
            alias=alias,
            core_number=core_number,
            expected_cells=expected_cells,
        )
        if loaded_core.h0.shape != core.h0.shape:
            raise EmbeddingClusterAnalysisError(f"Saved embedding shape drifted for {alias}.")
        relative = output_path.relative_to(output_root).as_posix()
        file_records[relative] = _file_record(output_path)
        core_receipts.append(
            {
                **expected,
                "embedding_file": relative,
                "embedding_file_sha256": file_records[relative]["sha256"],
                "cell_order": "prepared_core_row_order",
                "stable_cell_key": f"{alias}:zero_padded_cell_index",
                "coordinate_units": "micrometres",
                "coordinate_orientation": "repository_global_frame_invert_y_for_plotting",
                "mask_dtype": "bool",
                "mask_true_count": 0,
                "mask_shape": [expected_cells, batch.n_genes],
                "expression_pre_sha256": expression_before,
                "expression_post_sha256": expression_after,
                "metadata_pre_sha256": metadata_before,
                "metadata_post_sha256": metadata_after,
                "metadata_unchanged": True,
                "prediction_invariance": {
                    "allclose": True,
                    "exact_equal": prediction_exact,
                    "maximum_absolute_difference": prediction_maximum,
                    "relative_tolerance": PREDICTION_INVARIANCE_RTOL,
                    "absolute_tolerance": PREDICTION_INVARIANCE_ATOL,
                    "note": (
                        "The optional outputs do not enter the prediction path; "
                        "CUDA sparse reductions may vary at floating-point "
                        "roundoff scale across repeated forwards."
                    ),
                    "shape": list(ordinary.prediction.shape),
                },
                "array_shapes": {
                    "h0": list(core.h0.shape),
                    "hL": list(core.hL.shape),
                    "delta_h": list(core.delta_h.shape),
                    "delta_h_l2": list(core.delta_h_l2.shape),
                    "coordinates_um": list(core.coordinates_um.shape),
                },
                "array_checksums": {
                    "cell_index": _array_sha256("cell_index", core.cell_index),
                    "core_number": _array_sha256(
                        "core_number", np.asarray(core_number, dtype=np.int16)
                    ),
                    "coordinates_um": _array_sha256(
                        "coordinates_um", core.coordinates_um
                    ),
                    "h0": _array_sha256("h0", core.h0),
                    "hL": _array_sha256("hL", core.hL),
                    "delta_h": _array_sha256("delta_h", core.delta_h),
                    "delta_h_l2": _array_sha256(
                        "delta_h_l2", core.delta_h_l2
                    ),
                },
                "finite": True,
                "delta_norm_nonnegative": True,
            }
        )
        del ordinary, extended
        del expression, gene_mask, covariates, edge_index, relative_geometry
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    total_cells = sum(int(value["cell_count"]) for value in core_receipts)
    hidden_shapes = {
        tuple(value["array_shapes"]["h0"][1:]) for value in core_receipts
    }
    if total_cells != EXPECTED_TOTAL_CELLS or len(hidden_shapes) != 1:
        raise EmbeddingClusterAnalysisError("Final extraction coverage or width is invalid.")
    receipt = _receipt_with_self_hash(
        {
            "schema": EXTRACTION_SCHEMA,
            "status": "complete",
            "run_id": inputs.run_id,
            "model_seed": int(inputs.checkpoint_payload["model_seed"]),
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "core_order": list(CORE_NUMBERS),
            "alias_order": list(CANCER_ALIASES),
            "total_cells": total_cells,
            "embedding_dimension": int(next(iter(hidden_shapes))[0]),
            "inference": {
                "model_eval": True,
                "torch_inference_mode": True,
                "dropout_disabled": True,
                "attention_dropout": 0.0,
                "neighbor_sampling": False,
                "complete_core_at_a_time": True,
                "dtype": "float32",
                "device": str(resolved_device),
                "coordinates_supplied_to_node_encoder": False,
                "batch_correction_added": False,
            },
            "input_provenance": inputs.provenance,
            "cores": core_receipts,
            "files": file_records,
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_extraction_receipt(output_root, receipt, inputs=inputs)
    return receipt


def load_all_core_embeddings(
    *, output_root: Path, extraction_receipt: Mapping[str, Any]
) -> tuple[CoreEmbeddings, ...]:
    records = extraction_receipt.get("cores")
    if not isinstance(records, list) or tuple(
        int(record.get("core_number", -1))
        for record in records
        if isinstance(record, Mapping)
    ) != CORE_NUMBERS:
        raise EmbeddingClusterAnalysisError("Extraction core receipt order is invalid.")
    result: list[CoreEmbeddings] = []
    for record in records:
        assert isinstance(record, Mapping)
        result.append(
            load_core_embedding_file(
                output_root / str(record["embedding_file"]),
                alias=str(record["alias"]),
                core_number=int(record["core_number"]),
                expected_cells=int(record["cell_count"]),
            )
        )
    if tuple(core.alias for core in result) != CANCER_ALIASES:
        raise EmbeddingClusterAnalysisError("All six cores are not present in order.")
    return tuple(result)


def concatenate_representation(
    cores: Sequence[CoreEmbeddings], *, representation: str
) -> np.ndarray:
    if representation not in REPRESENTATIONS:
        raise EmbeddingClusterAnalysisError("Unknown embedding representation.")
    if tuple(core.core_number for core in cores) != CORE_NUMBERS:
        raise EmbeddingClusterAnalysisError(
            "Joint clustering requires all six cores in locked order."
        )
    name = REPRESENTATION_ARRAY[representation]
    arrays = [np.asarray(getattr(core, name), dtype=np.float32) for core in cores]
    if len({array.shape[1] for array in arrays}) != 1:
        raise EmbeddingClusterAnalysisError("Embedding widths differ across cores.")
    combined = np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)
    if not np.isfinite(combined).all():
        raise EmbeddingClusterAnalysisError("Joint embeddings contain non-finite values.")
    if not np.any(np.var(combined.astype(np.float64), axis=0) > 0.0):
        raise EmbeddingClusterAnalysisError("Joint embeddings have zero variance.")
    return combined


def deterministic_pca(
    embeddings: np.ndarray, *, n_components: int
) -> PCAResult:
    """Mean-center and exactly diagonalize the small feature covariance matrix."""

    values = np.asarray(embeddings)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 1
        or not np.isfinite(values).all()
    ):
        raise EmbeddingClusterAnalysisError("PCA input must be finite [cells, dims].")
    if isinstance(n_components, bool) or int(n_components) <= 0:
        raise EmbeddingClusterAnalysisError("PCA components must be positive.")
    retained = min(int(n_components), values.shape[0] - 1, values.shape[1])
    work = values.astype(np.float64, copy=True)
    mean = work.mean(axis=0, dtype=np.float64)
    work -= mean
    variances = np.var(work, axis=0, ddof=1)
    if not np.any(variances > 0.0):
        raise EmbeddingClusterAnalysisError("PCA input has zero total variance.")
    covariance = (work.T @ work) / float(values.shape[0] - 1)
    if covariance.shape != (values.shape[1], values.shape[1]):
        raise EmbeddingClusterAnalysisError("PCA covariance shape is invalid.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[:retained]]
    for column in range(components.shape[1]):
        pivot = int(np.argmax(np.abs(components[:, column])))
        if components[pivot, column] < 0.0:
            components[:, column] *= -1.0
    scores = work @ components
    row_norms = np.linalg.norm(scores, axis=1)
    if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= 0.0):
        raise EmbeddingClusterAnalysisError(
            "Retained PCA representation contains a zero-norm cell."
        )
    scores /= row_norms[:, None]
    normalized = np.ascontiguousarray(scores, dtype=np.float32)
    total_variance = float(eigenvalues.sum())
    explained = eigenvalues[:retained]
    ratios = explained / total_variance
    receipt = {
        "method": "exact_feature_covariance_eigendecomposition",
        "mean_centered": True,
        "input_shape": list(values.shape),
        "requested_components": int(n_components),
        "retained_components": retained,
        "feature_covariance_shape": list(covariance.shape),
        "cell_by_cell_matrix_constructed": False,
        "component_sign_rule": "largest_absolute_loading_positive",
        "l2_normalized_after_pca": True,
        "input_nonzero_variance_dimensions": int(np.count_nonzero(variances > 0.0)),
        "explained_variance": explained.tolist(),
        "explained_variance_ratio": ratios.tolist(),
        "total_explained_variance_ratio": float(ratios.sum()),
        "center_sha256": _array_sha256("pca_center", mean),
        "components_sha256": _array_sha256("pca_components", components),
        "normalized_scores_sha256": _array_sha256(
            "normalized_pca_scores", normalized
        ),
    }
    return PCAResult(normalized_scores=normalized, receipt=receipt)


def build_faiss_cosine_knn_graph(
    normalized_scores: np.ndarray,
    *,
    n_neighbors: int,
    random_seed: int,
) -> KNNGraphResult:
    """Construct a deterministic single-threaded sparse FAISS-HNSW kNN union."""

    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise EmbeddingClusterAnalysisError(
            "FAISS is required; install the embedding-analysis optional dependencies."
        ) from exc
    values = np.ascontiguousarray(normalized_scores, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise EmbeddingClusterAnalysisError("kNN input must be finite [cells, PCA].")
    n_cells, dimension = values.shape
    if (
        isinstance(n_neighbors, bool)
        or int(n_neighbors) <= 0
        or int(n_neighbors) >= n_cells
    ):
        raise EmbeddingClusterAnalysisError("n_neighbors must be in [1, cells-1].")
    norms = np.sqrt(
        np.einsum("ij,ij->i", values, values, dtype=np.float64, optimize=True)
    )
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise EmbeddingClusterAnalysisError("FAISS cosine input is not L2-normalized.")
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = max(HNSW_EF_SEARCH, int(n_neighbors) + 16)
    index.add(values)
    search_width = min(n_cells, int(n_neighbors) + 8)
    similarities, indices = index.search(values, search_width)
    selected = np.empty((n_cells, int(n_neighbors)), dtype=np.int64)
    selected_similarity = np.empty((n_cells, int(n_neighbors)), dtype=np.float32)
    for row in range(n_cells):
        candidates = indices[row]
        scores = similarities[row]
        valid = (candidates >= 0) & (candidates != row)
        candidates = candidates[valid]
        scores = scores[valid]
        if len(candidates) < int(n_neighbors):
            raise EmbeddingClusterAnalysisError(
                f"FAISS returned too few non-self neighbors for cell {row}."
            )
        # Deterministically order the approximate candidate set by decreasing
        # cosine similarity, then stable global cell index for ties.
        order = np.lexsort((candidates, -scores))[: int(n_neighbors)]
        chosen = candidates[order]
        if len(np.unique(chosen)) != int(n_neighbors):
            raise EmbeddingClusterAnalysisError("FAISS returned duplicate neighbors.")
        selected[row] = chosen
        selected_similarity[row] = scores[order]
    source = np.repeat(np.arange(n_cells, dtype=np.int64), int(n_neighbors))
    target = selected.reshape(-1)
    low = np.minimum(source, target)
    high = np.maximum(source, target)
    if np.any(low == high):
        raise EmbeddingClusterAnalysisError("Embedding kNN graph contains self edges.")
    codes = low * np.int64(n_cells) + high
    unique_codes = np.unique(codes)
    edge_pairs = np.column_stack(
        (unique_codes // np.int64(n_cells), unique_codes % np.int64(n_cells))
    ).astype(np.int64, copy=False)
    edge_pairs = np.ascontiguousarray(edge_pairs)
    if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
        raise EmbeddingClusterAnalysisError("Sparse kNN edge shape is invalid.")
    receipt = {
        "implementation": "faiss.IndexHNSWFlat",
        "faiss_version": getattr(faiss, "__version__", _package_version("faiss-cpu")),
        "distance_metric": "cosine_via_l2_normalized_inner_product",
        "approximate_neighbor_index": True,
        "single_threaded_index_and_search": True,
        "input_order": "locked_global_core_then_cell_index",
        "random_seed": int(random_seed),
        "random_seed_role": "Leiden; FAISS HNSW uses fixed library RNG with stable insertion order",
        "hnsw_m": HNSW_M,
        "ef_construction": HNSW_EF_CONSTRUCTION,
        "ef_search": int(index.hnsw.efSearch),
        "n_neighbors": int(n_neighbors),
        "directed_neighbor_entries": int(selected.size),
        "symmetrization": "undirected_union",
        "edge_weights": "unweighted",
        "undirected_edges": int(len(edge_pairs)),
        "neighbor_storage_shape": list(selected.shape),
        "cell_by_cell_matrix_constructed": False,
        "maximum_explicit_pairwise_array_shape": list(indices.shape),
        "neighbors_sha256": _array_sha256("knn_neighbors", selected),
        "neighbor_cosine_sha256": _array_sha256(
            "knn_neighbor_cosine", selected_similarity
        ),
        "undirected_edges_sha256": _array_sha256(
            "knn_undirected_edges", edge_pairs
        ),
    }
    return KNNGraphResult(
        edge_pairs=edge_pairs,
        receipt=receipt,
        directed_neighbors=np.ascontiguousarray(selected, dtype=np.int64),
    )


def run_seeded_leiden(
    graph: KNNGraphResult,
    *,
    n_cells: int,
    resolution: float,
    random_seed: int,
) -> LeidenResult:
    """Run seeded Leiden and relabel by descending size/minimum cell index."""

    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise EmbeddingClusterAnalysisError(
            "igraph and leidenalg are required for clustering."
        ) from exc
    if not math.isfinite(float(resolution)) or float(resolution) <= 0.0:
        raise EmbeddingClusterAnalysisError("Leiden resolution must be positive.")
    edge_pairs = np.asarray(graph.edge_pairs, dtype=np.int64)
    if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
        raise EmbeddingClusterAnalysisError("Leiden edges must be sparse pairs.")
    igraph_graph = ig.Graph(n=int(n_cells), edges=edge_pairs, directed=False)
    if igraph_graph.vcount() != int(n_cells) or igraph_graph.ecount() != len(edge_pairs):
        raise EmbeddingClusterAnalysisError("igraph changed sparse graph identity.")
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=float(resolution),
        n_iterations=-1,
        seed=int(random_seed),
    )
    raw_labels = np.asarray(partition.membership, dtype=np.int64)
    if raw_labels.shape != (int(n_cells),) or np.any(raw_labels < 0):
        raise EmbeddingClusterAnalysisError("Leiden returned invalid memberships.")
    raw_ids, counts = np.unique(raw_labels, return_counts=True)
    ordering = sorted(
        range(len(raw_ids)),
        key=lambda index: (
            -int(counts[index]),
            int(np.flatnonzero(raw_labels == raw_ids[index])[0]),
            int(raw_ids[index]),
        ),
    )
    mapping = {
        int(raw_ids[old_index]): new_index
        for new_index, old_index in enumerate(ordering)
    }
    labels = np.asarray([mapping[int(value)] for value in raw_labels], dtype=np.int64)
    sorted_sizes = np.bincount(labels)
    adjacency = coo_matrix(
        (
            np.ones(2 * len(edge_pairs), dtype=np.uint8),
            (
                np.concatenate((edge_pairs[:, 0], edge_pairs[:, 1])),
                np.concatenate((edge_pairs[:, 1], edge_pairs[:, 0])),
            ),
        ),
        shape=(int(n_cells), int(n_cells)),
    ).tocsr()
    component_count, component_labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    component_sizes = np.bincount(component_labels, minlength=component_count)
    quality = float(partition.quality())
    modularity = float(igraph_graph.modularity(labels.tolist()))
    receipt = {
        "implementation": "leidenalg.RBConfigurationVertexPartition",
        "igraph_version": getattr(ig, "__version__", _package_version("igraph")),
        "leidenalg_version": getattr(
            leidenalg, "__version__", _package_version("leidenalg")
        ),
        "resolution": float(resolution),
        "random_seed": int(random_seed),
        "n_iterations": -1,
        "quality": quality,
        "modularity": modularity,
        "number_of_graph_components": int(component_count),
        "largest_graph_component_size": int(component_sizes.max(initial=0)),
        "largest_graph_component_proportion": float(
            component_sizes.max(initial=0) / int(n_cells)
        ),
        "cluster_count": int(len(sorted_sizes)),
        "cluster_size_min": int(sorted_sizes.min()),
        "cluster_size_max": int(sorted_sizes.max()),
        "cluster_sort": "descending_size_then_minimum_global_cell_index_then_raw_id",
        "raw_to_sorted_cluster": {str(key): value for key, value in mapping.items()},
        "labels_sha256": _array_sha256("sorted_leiden_labels", labels),
    }
    return LeidenResult(labels=labels, receipt=receipt)


def cluster_joint_representation(
    cores: Sequence[CoreEmbeddings],
    *,
    representation: str,
    n_neighbors: int,
    leiden_resolution: float,
    pca_components: int,
    random_seed: int,
) -> tuple[LeidenResult, Mapping[str, Any]]:
    """Execute one complete, representation-specific PCA/kNN/Leiden pipeline."""

    combined = concatenate_representation(cores, representation=representation)
    pca = deterministic_pca(combined, n_components=pca_components)
    knn = build_faiss_cosine_knn_graph(
        pca.normalized_scores,
        n_neighbors=n_neighbors,
        random_seed=random_seed,
    )
    leiden = run_seeded_leiden(
        knn,
        n_cells=len(combined),
        resolution=leiden_resolution,
        random_seed=random_seed,
    )
    receipt = {
        "representation": representation,
        "embedding_array": REPRESENTATION_ARRAY[representation],
        "label_prefix": REPRESENTATION_PREFIX[representation],
        "joint_core_order": list(CORE_NUMBERS),
        "joint_cell_count": int(len(combined)),
        "joint_embedding_sha256": _array_sha256(
            f"joint_{representation}_embedding", combined
        ),
        "pipeline_instance_sha256": _canonical_sha256(
            {
                "representation": representation,
                "embedding": _array_sha256(
                    f"joint_{representation}_embedding", combined
                ),
                "pca": pca.receipt["normalized_scores_sha256"],
                "knn": knn.receipt["undirected_edges_sha256"],
            }
        ),
        "pca": pca.receipt,
        "knn": knn.receipt,
        "leiden": leiden.receipt,
    }
    return leiden, receipt


def build_cell_index_frame(cores: Sequence[CoreEmbeddings]) -> pd.DataFrame:
    """Build the one-row-per-cell non-identifying alignment table."""

    if tuple(core.core_number for core in cores) != CORE_NUMBERS:
        raise EmbeddingClusterAnalysisError("Cell table requires all six ordered cores.")
    frames: list[pd.DataFrame] = []
    global_start = 0
    for core in cores:
        expected = np.arange(core.n_cells, dtype=np.int64)
        if not np.array_equal(core.cell_index, expected):
            raise EmbeddingClusterAnalysisError(
                f"Cell indices are not canonical for {core.alias}."
            )
        frame = pd.DataFrame(
            {
                "global_cell_index": np.arange(
                    global_start, global_start + core.n_cells, dtype=np.int64
                ),
                "cell_index": core.cell_index,
                "cell_key": [
                    f"{core.alias}:{int(index):08d}" for index in core.cell_index
                ],
                "core_alias": core.alias,
                "core_number": np.full(core.n_cells, core.core_number, dtype=np.int16),
                "x_um": core.coordinates_um[:, 0],
                "y_um": core.coordinates_um[:, 1],
                "delta_h_l2": core.delta_h_l2,
            }
        )
        frames.append(frame)
        global_start += core.n_cells
    result = pd.concat(frames, ignore_index=True)
    if (
        len(result) != sum(core.n_cells for core in cores)
        or result["cell_key"].duplicated().any()
        or not np.array_equal(
            result["global_cell_index"].to_numpy(),
            np.arange(len(result), dtype=np.int64),
        )
        or tuple(result["core_number"].drop_duplicates().tolist()) != CORE_NUMBERS
        or not np.isfinite(result[["x_um", "y_um", "delta_h_l2"]].to_numpy()).all()
    ):
        raise EmbeddingClusterAnalysisError("Cell alignment table is invalid.")
    return result


def cluster_summary_tables(
    labels: np.ndarray,
    cell_frame: pd.DataFrame,
    *,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return deterministic cluster and six-core composition summaries."""

    memberships = np.asarray(labels, dtype=np.int64)
    if memberships.shape != (len(cell_frame),) or np.any(memberships < 0):
        raise EmbeddingClusterAnalysisError("Cluster labels do not align with cells.")
    unique = np.unique(memberships)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise EmbeddingClusterAnalysisError("Cluster labels are not contiguous.")
    core_values = cell_frame["core_number"].to_numpy(dtype=np.int64)
    core_totals = {
        core: int(np.count_nonzero(core_values == core)) for core in CORE_NUMBERS
    }
    summary_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    dominated: list[str] = []
    for cluster_id in unique:
        selected = memberships == cluster_id
        cluster_size = int(np.count_nonzero(selected))
        label = f"{prefix}{int(cluster_id)}"
        counts = {
            core: int(np.count_nonzero(selected & (core_values == core)))
            for core in CORE_NUMBERS
        }
        dominant_core = min(
            CORE_NUMBERS, key=lambda core: (-counts[core], CORE_NUMBERS.index(core))
        )
        dominant_proportion = counts[dominant_core] / cluster_size
        is_dominated = bool(dominant_proportion > 0.90)
        if is_dominated:
            dominated.append(label)
        summary_rows.append(
            {
                "cluster": label,
                "cluster_number": int(cluster_id),
                "size": cluster_size,
                "proportion": cluster_size / len(memberships),
                "dominant_core": int(dominant_core),
                "dominant_core_count": counts[dominant_core],
                "dominant_core_proportion": dominant_proportion,
                "core_dominated_gt_90pct": is_dominated,
            }
        )
        for core in CORE_NUMBERS:
            composition_rows.append(
                {
                    "cluster": label,
                    "cluster_number": int(cluster_id),
                    "core_number": int(core),
                    "cell_count": counts[core],
                    "proportion_within_cluster": counts[core] / cluster_size,
                    "proportion_within_core": counts[core] / core_totals[core],
                    "cluster_size": cluster_size,
                    "core_dominated_gt_90pct": is_dominated,
                }
            )
    summary = pd.DataFrame(summary_rows)
    composition = pd.DataFrame(composition_rows)
    if int(summary["size"].sum()) != len(memberships):
        raise EmbeddingClusterAnalysisError("Cluster summary dropped cells.")
    return summary, composition, dominated


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ np.asarray(
        [
            [0.4124564, 0.2126729, 0.0193339],
            [0.3575761, 0.7151522, 0.1191920],
            [0.1804375, 0.0721750, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz /= np.asarray([0.95047, 1.0, 1.08883], dtype=np.float64)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.column_stack(
        (
            116.0 * transformed[:, 1] - 16.0,
            500.0 * (transformed[:, 0] - transformed[:, 1]),
            200.0 * (transformed[:, 1] - transformed[:, 2]),
        )
    )


def deterministic_glasbey_palette(
    cluster_count: int, *, namespace: str
) -> dict[str, str]:
    """Create a deterministic greedy farthest-point palette in CIELAB space."""

    if namespace not in REPRESENTATIONS:
        raise EmbeddingClusterAnalysisError("Palette namespace is invalid.")
    if isinstance(cluster_count, bool) or int(cluster_count) <= 0:
        raise EmbeddingClusterAnalysisError("Palette size must be positive.")
    levels = np.linspace(0.04, 0.96, 16, dtype=np.float64)
    candidates = np.stack(
        np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    lab = _srgb_to_lab(candidates)
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    keep = (lab[:, 0] >= 25.0) & (lab[:, 0] <= 88.0) & (chroma >= 20.0)
    candidates = candidates[keep]
    lab = lab[keep]
    anchor_hex = "#0072B2" if namespace == "intrinsic" else "#D55E00"
    anchor_rgb = np.asarray(
        [[int(anchor_hex[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]],
        dtype=np.float64,
    )
    anchor_lab = _srgb_to_lab(anchor_rgb)[0]
    minimum_distance = np.sum((lab - anchor_lab) ** 2, axis=1)
    selected_rgb = [anchor_rgb[0]]
    # A namespace-specific stable candidate rotation prevents the two palettes
    # from implying a cross-representation cluster correspondence.
    rotation = 0 if namespace == "intrinsic" else len(candidates) // 3
    candidate_order = np.roll(np.arange(len(candidates), dtype=np.int64), rotation)
    available = np.ones(len(candidates), dtype=bool)
    for _ in range(1, int(cluster_count)):
        available_scores = np.where(available, minimum_distance, -1.0)
        maximum = float(available_scores.max())
        ties = np.flatnonzero(np.isclose(available_scores, maximum, rtol=0.0, atol=1e-12))
        if not len(ties):
            raise EmbeddingClusterAnalysisError("Palette candidate space was exhausted.")
        tie_ranks = {int(value): rank for rank, value in enumerate(candidate_order)}
        selected = min((int(value) for value in ties), key=tie_ranks.__getitem__)
        available[selected] = False
        selected_rgb.append(candidates[selected])
        distance = np.sum((lab - lab[selected]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)

    prefix = REPRESENTATION_PREFIX[namespace]
    palette: dict[str, str] = {}
    for index, rgb in enumerate(selected_rgb):
        channels = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
        palette[f"{prefix}{index}"] = "#" + "".join(
            f"{int(channel):02X}" for channel in channels
        )
    if len(set(palette.values())) != int(cluster_count):
        raise EmbeddingClusterAnalysisError("Palette contains duplicate colors.")
    return palette


def delta_norm_statistics(cores: Sequence[CoreEmbeddings]) -> dict[str, Any]:
    values = np.concatenate([core.delta_h_l2 for core in cores]).astype(
        np.float64, copy=False
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise EmbeddingClusterAnalysisError("Delta norms are invalid.")
    p01, p99 = np.quantile(values, [0.01, 0.99])
    per_core: list[dict[str, Any]] = []
    for core in cores:
        per_core.append(
            {
                "core_number": core.core_number,
                "count": core.n_cells,
                "minimum": float(core.delta_h_l2.min()),
                "maximum": float(core.delta_h_l2.max()),
                "mean": float(core.delta_h_l2.mean()),
                "median": float(np.median(core.delta_h_l2)),
                "standard_deviation": float(core.delta_h_l2.std(ddof=0)),
            }
        )
    return {
        "description": "contextual representation-change magnitude",
        "biological_influence_or_causal_effect": False,
        "count": int(len(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=0)),
        "p01": float(p01),
        "p99": float(p99),
        "plotting_limits": {
            "vmin_global_p01": float(p01),
            "vmax_global_p99": float(p99),
            "shared_across_all_six_cores": True,
            "raw_values_clipped_in_saved_data": False,
        },
        "per_core": per_core,
    }


def _clustering_configuration(
    *,
    inputs: ResolvedAnalysisInputs,
    extraction_receipt: Mapping[str, Any],
    n_neighbors: int,
    leiden_resolution: float,
    pca_components: int,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "run_id": inputs.run_id,
        "checkpoint_sha256": inputs.checkpoint_sha256,
        "extraction_manifest_sha256": _canonical_sha256(extraction_receipt),
        "core_order": list(CORE_NUMBERS),
        "joint_clustering": True,
        "n_neighbors": int(n_neighbors),
        "distance_metric": "cosine",
        "leiden_resolution": float(leiden_resolution),
        "pca_components": int(pca_components),
        "random_seed": int(random_seed),
    }


def _verify_clustering_receipt(
    output_root: Path,
    receipt: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label="clustering receipt")
    if (
        receipt.get("schema") != CLUSTERING_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("configuration_sha256") != _canonical_sha256(configuration)
        or tuple(receipt.get("core_order", ())) != CORE_NUMBERS
        or receipt.get("independent_representation_pipelines") is not True
    ):
        raise EmbeddingClusterAnalysisError("Clustering receipt identity is invalid.")
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise EmbeddingClusterAnalysisError("Clustering receipt lacks files.")
    _verify_stage_files(output_root, files, label="clustering")


def build_clustering_outputs(
    *,
    inputs: ResolvedAnalysisInputs,
    output_root: Path,
    extraction_receipt: Mapping[str, Any],
    n_neighbors: int,
    leiden_resolution: float,
    pca_components: int,
    random_seed: int,
) -> Mapping[str, Any]:
    """Run or reuse two independent joint six-core clustering pipelines."""

    configuration = _clustering_configuration(
        inputs=inputs,
        extraction_receipt=extraction_receipt,
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        pca_components=pca_components,
        random_seed=random_seed,
    )
    receipt_path = output_root / "clustering" / "clustering_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="clustering receipt")
        _verify_clustering_receipt(
            output_root, receipt, configuration=configuration
        )
        return receipt
    clustering_dir = output_root / "clustering"
    tables_dir = output_root / "tables"
    if (
        (clustering_dir.exists() and any(clustering_dir.iterdir()))
        or (tables_dir.exists() and any(tables_dir.iterdir()))
    ):
        raise EmbeddingClusterAnalysisError(
            "Partial clustering/table outputs exist without a complete receipt."
        )
    clustering_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    cores = load_all_core_embeddings(
        output_root=output_root, extraction_receipt=extraction_receipt
    )
    cell_frame = build_cell_index_frame(cores)

    results: dict[str, LeidenResult] = {}
    pipeline_receipts: dict[str, Mapping[str, Any]] = {}
    for representation in REPRESENTATIONS:
        result, pipeline = cluster_joint_representation(
            cores,
            representation=representation,
            n_neighbors=n_neighbors,
            leiden_resolution=leiden_resolution,
            pca_components=pca_components,
            random_seed=random_seed,
        )
        results[representation] = result
        pipeline_receipts[representation] = pipeline
    if (
        pipeline_receipts["intrinsic"]["pipeline_instance_sha256"]
        == pipeline_receipts["contextual"]["pipeline_instance_sha256"]
    ):
        raise EmbeddingClusterAnalysisError(
            "Intrinsic and contextual pipelines did not retain separate identities."
        )

    stage_files: dict[str, dict[str, Any]] = {}
    dominated_by_representation: dict[str, list[str]] = {}
    cluster_size_ranges: dict[str, list[int]] = {}
    palettes: dict[str, Mapping[str, str]] = {}
    for representation in REPRESENTATIONS:
        labels = results[representation].labels
        prefix = REPRESENTATION_PREFIX[representation]
        id_column = f"{representation}_cluster_number"
        label_column = f"{representation}_cluster"
        cell_frame[id_column] = labels.astype(np.int32)
        cell_frame[label_column] = [f"{prefix}{int(value)}" for value in labels]
        summary, composition, dominated = cluster_summary_tables(
            labels, cell_frame, prefix=prefix
        )
        dominated_by_representation[representation] = dominated
        cluster_size_ranges[representation] = [
            int(summary["size"].min()),
            int(summary["size"].max()),
        ]
        palette = deterministic_glasbey_palette(
            len(summary), namespace=representation
        )
        palettes[representation] = palette
        label_path = clustering_dir / f"{representation}_labels.npy"
        parameter_path = clustering_dir / f"{representation}_clustering_parameters.json"
        palette_path = clustering_dir / f"{representation}_palette.json"
        summary_path = tables_dir / f"{representation}_cluster_summary.csv"
        composition_path = (
            tables_dir / f"{representation}_cluster_core_composition.csv"
        )
        _atomic_write_npy(label_path, labels)
        _atomic_write_json(parameter_path, pipeline_receipts[representation])
        _atomic_write_json(
            palette_path,
            {
                "representation": representation,
                "label_prefix": prefix,
                "method": "deterministic_greedy_farthest_point_CIELAB",
                "colors": palette,
            },
        )
        _atomic_write_csv(summary_path, summary)
        _atomic_write_csv(composition_path, composition)
        for path in (
            label_path,
            parameter_path,
            palette_path,
            summary_path,
            composition_path,
        ):
            stage_files[path.relative_to(output_root).as_posix()] = _file_record(path)

    table_columns = [
        "global_cell_index",
        "cell_index",
        "cell_key",
        "core_alias",
        "core_number",
        "x_um",
        "y_um",
        "delta_h_l2",
        "intrinsic_cluster_number",
        "intrinsic_cluster",
        "contextual_cluster_number",
        "contextual_cluster",
    ]
    cell_frame = cell_frame.loc[:, table_columns]
    table_path = tables_dir / "cell_embedding_clusters.parquet"
    _atomic_write_parquet(table_path, cell_frame)
    stage_files[table_path.relative_to(output_root).as_posix()] = _file_record(table_path)
    delta_statistics = delta_norm_statistics(cores)
    receipt = _receipt_with_self_hash(
        {
            "schema": CLUSTERING_SCHEMA,
            "status": "complete",
            "run_id": inputs.run_id,
            "configuration": configuration,
            "configuration_sha256": _canonical_sha256(configuration),
            "core_order": list(CORE_NUMBERS),
            "total_cells": int(len(cell_frame)),
            "independent_representation_pipelines": True,
            "spatial_training_graph_reused_for_clustering": False,
            "cross_core_embedding_neighbors_permitted": True,
            "pipelines": pipeline_receipts,
            "cluster_counts": {
                representation: int(
                    pipeline_receipts[representation]["leiden"]["cluster_count"]
                )
                for representation in REPRESENTATIONS
            },
            "cluster_size_ranges": cluster_size_ranges,
            "core_dominated_gt_90pct": dominated_by_representation,
            "palettes": palettes,
            "delta_h_l2": delta_statistics,
            "cell_table": {
                "row_count": int(len(cell_frame)),
                "columns": table_columns,
                "row_order": "core_1_9_13_15_21_23_then_prepared_cell_index",
                "stable_nonidentifying_cell_key": True,
            },
            "files": stage_files,
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_clustering_receipt(output_root, receipt, configuration=configuration)
    return receipt


def requested_panel_order() -> tuple[int, ...]:
    return CORE_NUMBERS


def spatial_plot_spec(
    *,
    intrinsic_palette: Mapping[str, str],
    contextual_palette: Mapping[str, str],
    delta_vmin: float,
    delta_vmax: float,
) -> dict[str, Any]:
    if not math.isfinite(delta_vmin) or not math.isfinite(delta_vmax) or delta_vmin >= delta_vmax:
        raise EmbeddingClusterAnalysisError("Delta plotting limits are invalid.")
    return {
        "panel_order": list(CORE_NUMBERS),
        "grid_shape": [2, 3],
        "equal_aspect": True,
        "invert_y_axis": True,
        "coordinate_units": "micrometres",
        "intrinsic_palette": dict(intrinsic_palette),
        "contextual_palette": dict(contextual_palette),
        "palettes_are_separate": dict(intrinsic_palette) != dict(contextual_palette),
        "delta_common_scale": {
            "vmin": float(delta_vmin),
            "vmax": float(delta_vmax),
            "shared_across_all_panels": True,
            "colormap": "viridis",
        },
    }


def _nice_scale_bar_length(coordinates: np.ndarray) -> float:
    x_range = float(np.ptp(coordinates[:, 0]))
    if not math.isfinite(x_range) or x_range <= 0.0:
        raise EmbeddingClusterAnalysisError("Cannot derive a physical scale bar.")
    target = x_range * 0.20
    exponent = 10.0 ** math.floor(math.log10(target))
    choices = [exponent, 2.0 * exponent, 5.0 * exponent, 10.0 * exponent]
    return max(value for value in choices if value <= target)


def _style_spatial_axis(axis: Any, coordinates: np.ndarray) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.invert_yaxis()
    axis.set_xlabel("x (µm)")
    axis.set_ylabel("y (µm)")
    axis.set_facecolor("#F8FAFC")
    axis.grid(False)
    length = _nice_scale_bar_length(coordinates)
    x_min, x_max = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
    y_min, y_max = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])
    x_range = float(x_max - x_min)
    y_range = float(y_max - y_min)
    x_start = float(x_min + 0.06 * x_range)
    y_value = float(y_max - 0.06 * y_range)
    axis.plot(
        [x_start, x_start + length],
        [y_value, y_value],
        color="#111827",
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=4,
    )
    axis.text(
        x_start + length / 2.0,
        y_value - 0.025 * y_range,
        f"{length:g} µm",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#111827",
        zorder=4,
    )


def _atomic_save_figure_pair(
    figure: Any,
    *,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    producer: str = "spatial_benchmark.relative_qkv_embedding_clustering",
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        for target, format_name, metadata in (
            (
                png_path,
                "png",
                {"Software": str(producer)},
            ),
            (
                pdf_path,
                "pdf",
                {
                    "Creator": str(producer),
                    "CreationDate": None,
                    "ModDate": None,
                },
            ),
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.tmp-",
                suffix=f".{format_name}",
                dir=target.parent,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            figure.savefig(
                temporary,
                format=format_name,
                dpi=dpi if format_name == "png" else None,
                bbox_inches="tight",
                facecolor="white",
                metadata=metadata,
            )
            os.replace(temporary, target)
            temporary_paths.remove(temporary)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def _cluster_legend_handles(palette: Mapping[str, str]) -> list[Any]:
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in sorted(
            palette.items(), key=lambda item: int(item[0][1:])
        )
    ]


def _render_combined_cluster_map(
    frame: pd.DataFrame,
    *,
    representation: str,
    palette: Mapping[str, str],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    label_column = f"{representation}_cluster"
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 11.5))
    for axis, core_number in zip(axes.ravel(), CORE_NUMBERS, strict=True):
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        colors = selected[label_column].map(palette)
        if colors.isna().any() or len(selected) == 0:
            raise EmbeddingClusterAnalysisError("Cluster palette or core panel is incomplete.")
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=0.75,
            c=colors.tolist(),
            marker="o",
            linewidths=0,
            edgecolors="none",
            alpha=0.90,
            rasterized=True,
        )
        axis.set_title(f"Cancer Core {core_number} (n={len(selected):,})", weight="bold")
        _style_spatial_axis(axis, coordinates)
    title = "Intrinsic h0 Leiden clusters" if representation == "intrinsic" else "Contextualized hL Leiden clusters"
    figure.suptitle(title + " — joint six-core clustering", fontsize=15, weight="bold")
    handles = _cluster_legend_handles(palette)
    columns = 2 if len(handles) > 24 else 1
    figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.865, 0.5),
        frameon=False,
        ncol=columns,
        title="Model-derived cluster",
        markerscale=2.0,
    )
    figure.subplots_adjust(left=0.06, right=0.85, bottom=0.07, top=0.92, wspace=0.25, hspace=0.27)
    _atomic_save_figure_pair(figure, png_path=png_path, pdf_path=pdf_path, dpi=dpi)
    plt.close(figure)


def _render_individual_cluster_maps(
    frame: pd.DataFrame,
    *,
    representation: str,
    palette: Mapping[str, str],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    label_column = f"{representation}_cluster"
    for core_number in CORE_NUMBERS:
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        colors = selected[label_column].map(palette)
        figure, axis = plt.subplots(figsize=(10.5, 9.0))
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=1.5,
            c=colors.tolist(),
            marker="o",
            linewidths=0,
            edgecolors="none",
            alpha=0.92,
            rasterized=True,
        )
        axis.set_title(
            f"Cancer Core {core_number} — {representation} joint Leiden clusters",
            weight="bold",
        )
        _style_spatial_axis(axis, coordinates)
        axis.legend(
            handles=_cluster_legend_handles(palette),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            ncol=2 if len(palette) > 24 else 1,
            title="Model-derived cluster",
        )
        figure.subplots_adjust(right=0.78)
        stem = f"{representation}_leiden_spatial_core_{core_number}"
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        _atomic_save_figure_pair(
            figure, png_path=png_path, pdf_path=pdf_path, dpi=dpi
        )
        plt.close(figure)
        outputs.extend((png_path, pdf_path))
    return outputs


def _render_combined_delta_map(
    frame: pd.DataFrame,
    *,
    vmin: float,
    vmax: float,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    normalization = Normalize(vmin=vmin, vmax=vmax, clip=True)
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 11.5))
    mappable = None
    for axis, core_number in zip(axes.ravel(), CORE_NUMBERS, strict=True):
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        mappable = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=0.80,
            c=selected["delta_h_l2"].to_numpy(dtype=np.float64),
            cmap="viridis",
            norm=normalization,
            marker="o",
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        axis.set_title(f"Cancer Core {core_number} (n={len(selected):,})", weight="bold")
        _style_spatial_axis(axis, coordinates)
    assert mappable is not None
    colorbar = figure.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        fraction=0.022,
        pad=0.02,
    )
    colorbar.set_label("||h_contextual − h_intrinsic||₂")
    figure.suptitle(
        "Contextual representation-change magnitude — shared global p1–p99 scale",
        fontsize=15,
        weight="bold",
    )
    figure.subplots_adjust(left=0.06, right=0.90, bottom=0.07, top=0.92, wspace=0.25, hspace=0.27)
    _atomic_save_figure_pair(figure, png_path=png_path, pdf_path=pdf_path, dpi=dpi)
    plt.close(figure)


def _render_individual_delta_maps(
    frame: pd.DataFrame,
    *,
    vmin: float,
    vmax: float,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    normalization = Normalize(vmin=vmin, vmax=vmax, clip=True)
    outputs: list[Path] = []
    for core_number in CORE_NUMBERS:
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        figure, axis = plt.subplots(figsize=(10.5, 9.0))
        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=1.5,
            c=selected["delta_h_l2"].to_numpy(dtype=np.float64),
            cmap="viridis",
            norm=normalization,
            marker="o",
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        axis.set_title(
            f"Cancer Core {core_number} — contextual representation-change magnitude",
            weight="bold",
        )
        _style_spatial_axis(axis, coordinates)
        colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_label("||h_contextual − h_intrinsic||₂")
        stem = f"delta_h_l2_spatial_core_{core_number}"
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        _atomic_save_figure_pair(
            figure, png_path=png_path, pdf_path=pdf_path, dpi=dpi
        )
        plt.close(figure)
        outputs.extend((png_path, pdf_path))
    return outputs


def render_spatial_outputs(
    *, output_root: Path, clustering_receipt: Mapping[str, Any], dpi: int = 300
) -> Mapping[str, Any]:
    """Render all combined and per-core tissue-coordinate maps."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 140,
            "savefig.dpi": int(dpi),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    table_path = output_root / "tables" / "cell_embedding_clusters.parquet"
    frame = pd.read_parquet(table_path)
    if tuple(frame["core_number"].drop_duplicates().tolist()) != CORE_NUMBERS:
        raise EmbeddingClusterAnalysisError("Plot table does not contain six ordered cores.")
    palettes = clustering_receipt.get("palettes")
    delta = clustering_receipt.get("delta_h_l2")
    if not isinstance(palettes, Mapping) or not isinstance(delta, Mapping):
        raise EmbeddingClusterAnalysisError("Clustering receipt lacks plotting metadata.")
    intrinsic_palette = dict(palettes["intrinsic"])
    contextual_palette = dict(palettes["contextual"])
    limits = delta.get("plotting_limits")
    if not isinstance(limits, Mapping):
        raise EmbeddingClusterAnalysisError("Delta plotting limits are missing.")
    vmin = float(limits["vmin_global_p01"])
    vmax = float(limits["vmax_global_p99"])
    specification = spatial_plot_spec(
        intrinsic_palette=intrinsic_palette,
        contextual_palette=contextual_palette,
        delta_vmin=vmin,
        delta_vmax=vmax,
    )
    figure_dir = output_root / "figures"
    per_core_dir = figure_dir / "per_core"
    per_core_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for representation, palette in (
        ("intrinsic", intrinsic_palette),
        ("contextual", contextual_palette),
    ):
        combined_png = figure_dir / f"{representation}_leiden_spatial_6cores.png"
        combined_pdf = figure_dir / f"{representation}_leiden_spatial_6cores.pdf"
        _render_combined_cluster_map(
            frame,
            representation=representation,
            palette=palette,
            png_path=combined_png,
            pdf_path=combined_pdf,
            dpi=dpi,
        )
        outputs.extend((combined_png, combined_pdf))
        outputs.extend(
            _render_individual_cluster_maps(
                frame,
                representation=representation,
                palette=palette,
                output_dir=per_core_dir,
                dpi=dpi,
            )
        )
    delta_png = figure_dir / "delta_h_l2_spatial_6cores.png"
    delta_pdf = figure_dir / "delta_h_l2_spatial_6cores.pdf"
    _render_combined_delta_map(
        frame,
        vmin=vmin,
        vmax=vmax,
        png_path=delta_png,
        pdf_path=delta_pdf,
        dpi=dpi,
    )
    outputs.extend((delta_png, delta_pdf))
    outputs.extend(
        _render_individual_delta_maps(
            frame,
            vmin=vmin,
            vmax=vmax,
            output_dir=per_core_dir,
            dpi=dpi,
        )
    )
    expected_count = 6 + (len(CORE_NUMBERS) * 2 * 3)
    if len(outputs) != expected_count or any(not path.is_file() for path in outputs):
        raise EmbeddingClusterAnalysisError("Requested spatial figures are incomplete.")
    return {
        "dpi": int(dpi),
        "combined_figure_count": 6,
        "per_core_figure_count": len(CORE_NUMBERS) * 2 * 3,
        "point_layer_rasterized_in_pdf": True,
        "one_dot_per_cell": True,
        "marker_borders": False,
        "lines_between_cells": False,
        "plot_specification": specification,
        "files": {
            path.relative_to(output_root).as_posix(): _file_record(path)
            for path in outputs
        },
    }


def _render_readme(
    *,
    inputs: ResolvedAnalysisInputs,
    extraction: Mapping[str, Any],
    clustering: Mapping[str, Any],
    output_root: Path,
) -> str:
    checkpoint_relative = (
        inputs.checkpoint_path.relative_to(inputs.project_root).as_posix()
        if inputs.checkpoint_path.is_relative_to(inputs.project_root)
        else inputs.checkpoint_path.as_posix()
    )
    output_relative = (
        output_root.relative_to(inputs.project_root).as_posix()
        if output_root.is_relative_to(inputs.project_root)
        else output_root.as_posix()
    )
    counts = ", ".join(
        f"Core {record['core_number']}: {int(record['cell_count']):,}"
        for record in extraction["cores"]
    )
    dominated_i = ", ".join(clustering["core_dominated_gt_90pct"]["intrinsic"]) or "none"
    dominated_c = ", ".join(clustering["core_dominated_gt_90pct"]["contextual"]) or "none"
    command = (
        "PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \\\n"
        "  analyze-embedding-clusters \\\n"
        f"  --run-id {inputs.run_id} \\\n"
        f"  --checkpoint {shlex.quote(checkpoint_relative)} \\\n"
        f"  --n-neighbors {clustering['configuration']['n_neighbors']} \\\n"
        f"  --leiden-resolution {clustering['configuration']['leiden_resolution']} \\\n"
        f"  --pca-components {clustering['configuration']['pca_components']} \\\n"
        f"  --random-seed {clustering['configuration']['random_seed']} \\\n"
        f"  --device {shlex.quote(str(extraction['inference']['device']))} \\\n"
        f"  --output-dir {shlex.quote(output_relative)}"
    )
    delta = clustering["delta_h_l2"]
    return f"""# Six-core intrinsic/contextual embedding clustering

Status: complete exploratory post-hoc analysis of a locked, completed model.
No model retraining, preprocessing refit, batch correction, neighbor sampling,
or per-core clustering was performed.

## Selected model and coverage

- Run ID: `{inputs.run_id}`
- Model seed: `{inputs.checkpoint_payload['model_seed']}`
- Checkpoint: `{checkpoint_relative}`
- Checkpoint SHA-256: `{inputs.checkpoint_sha256}`
- Model width/layers: `{extraction['embedding_dimension']}` / `{inputs.checkpoint_payload['model_construction']['graph_layers']}`
- Cells: {counts}; total {extraction['total_cells']:,}
- Joint intrinsic clusters: {clustering['cluster_counts']['intrinsic']}
- Joint contextual clusters: {clustering['cluster_counts']['contextual']}
- Intrinsic clusters >90% from one core: {dominated_i}
- Contextual clusters >90% from one core: {dominated_c}
- Global raw delta-norm mean/median: {delta['mean']:.6g} / {delta['median']:.6g}
- Shared plotting limits (global p1, p99): {delta['p01']:.6g}, {delta['p99']:.6g}

The core counts differ materially from the approximate 15,000-per-core planning
expectation. The immutable preparation manifest contains 117,996 cells, and all
117,996 are retained here without downsampling or silent exclusion.

## Interpretation constraints

Intrinsic clusters represent patterns in the cell's own expression and metadata
embedding. Contextual clusters represent patterns after graph-based neighborhood
processing. Delta-h norm measures the magnitude of representation change after
contextual processing and is only a descriptive contextual
representation-change magnitude.

None of these quantities independently establishes cell type, signaling,
biological influence, or causality. The model-derived clusters are deliberately
not assigned biological names. Marker-based and pathological validation will be
conducted separately.

This analysis is transductive and post-hoc. Core-dominated clusters are flagged,
not removed or integrated. No Harmony, scVI, ComBat, or related integration was
applied.

## Reproduction

From the repository root:

```bash
{command}
```

The workflow is resumable: a checksum-valid
`embeddings/extraction_manifest.json` skips model inference, and a checksum-valid
`clustering/clustering_manifest.json` skips clustering when only figures remain.
All final files are checksum-bound by `manifest.json`.
"""


def _required_combined_figures() -> tuple[str, ...]:
    return (
        "figures/intrinsic_leiden_spatial_6cores.png",
        "figures/intrinsic_leiden_spatial_6cores.pdf",
        "figures/contextual_leiden_spatial_6cores.png",
        "figures/contextual_leiden_spatial_6cores.pdf",
        "figures/delta_h_l2_spatial_6cores.png",
        "figures/delta_h_l2_spatial_6cores.pdf",
    )


def _required_per_core_figures() -> tuple[str, ...]:
    files: list[str] = []
    for core_number in CORE_NUMBERS:
        for representation in REPRESENTATIONS:
            stem = f"{representation}_leiden_spatial_core_{core_number}"
            files.extend(
                f"figures/per_core/{stem}.{suffix}"
                for suffix in ("png", "pdf")
            )
        delta_stem = f"delta_h_l2_spatial_core_{core_number}"
        files.extend(
            f"figures/per_core/{delta_stem}.{suffix}"
            for suffix in ("png", "pdf")
        )
    return tuple(files)


def _required_analysis_files() -> set[str]:
    return {
        *_required_combined_figures(),
        *_required_per_core_figures(),
        *(f"embeddings/core_{core}_embeddings.npz" for core in CORE_NUMBERS),
        "embeddings/extraction_manifest.json",
        "tables/cell_embedding_clusters.parquet",
        "tables/intrinsic_cluster_summary.csv",
        "tables/contextual_cluster_summary.csv",
        "tables/intrinsic_cluster_core_composition.csv",
        "tables/contextual_cluster_core_composition.csv",
        "clustering/clustering_manifest.json",
        "clustering/intrinsic_labels.npy",
        "clustering/contextual_labels.npy",
        "clustering/intrinsic_clustering_parameters.json",
        "clustering/contextual_clustering_parameters.json",
        "clustering/intrinsic_palette.json",
        "clustering/contextual_palette.json",
        "README.md",
    }


def _is_hex_digest(value: object, *, length: int) -> bool:
    text_value = str(value)
    return len(text_value) == length and all(
        character in "0123456789abcdef" for character in text_value
    )


def _is_sha256(value: object) -> bool:
    return _is_hex_digest(value, length=64)


def _verify_final_semantics(
    root: Path,
    manifest: Mapping[str, Any],
    files: Mapping[str, Any],
) -> None:
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("run_id") != EXPECTED_RUN_ID
        or int(manifest.get("model_seed", -1)) != EXPECTED_MODEL_SEED
        or not _is_sha256(manifest.get("checkpoint_sha256"))
        or not _is_sha256(manifest.get("preprocessing_checksum"))
        or not _is_hex_digest(manifest.get("training_source_commit"), length=40)
        or not _is_hex_digest(manifest.get("analysis_source_commit"), length=40)
    ):
        raise EmbeddingClusterAnalysisError("Final analysis provenance is incomplete.")
    construction = manifest.get("model_construction")
    if not isinstance(construction, Mapping) or (
        int(construction.get("num_genes", -1)) != 1_000
        or int(construction.get("node_covariate_dim", -1)) != 22
        or int(construction.get("hidden_dim", -1)) != 256
        or int(construction.get("embedding_dim", -1)) != 256
        or int(construction.get("graph_layers", -1)) != 4
        or int(construction.get("relative_geometry_dim", -1)) != 70
    ):
        raise EmbeddingClusterAnalysisError("Final model dimensions are incomplete.")
    cell_counts = manifest.get("cell_counts")
    if not isinstance(cell_counts, Mapping) or {
        int(core): int(count) for core, count in cell_counts.items()
    } != EXPECTED_CELL_COUNTS:
        raise EmbeddingClusterAnalysisError("Final cell counts changed.")
    shapes = manifest.get("embedding_shapes")
    if not isinstance(shapes, Mapping):
        raise EmbeddingClusterAnalysisError("Final embedding shapes are missing.")
    for core_number, cell_count in EXPECTED_CELL_COUNTS.items():
        record = shapes.get(str(core_number))
        expected_shape = [cell_count, 256]
        if not isinstance(record, Mapping) or (
            record.get("h0") != expected_shape or record.get("hL") != expected_shape
        ):
            raise EmbeddingClusterAnalysisError("Final embedding shapes changed.")

    source_cores = manifest.get("source_core_artifacts")
    if not isinstance(source_cores, list) or tuple(
        int(record.get("core_number", -1))
        for record in source_cores
        if isinstance(record, Mapping)
    ) != CORE_NUMBERS:
        raise EmbeddingClusterAnalysisError("Source-core provenance is incomplete.")
    if any(
        not _is_sha256(record.get(name))
        for record in source_cores
        if isinstance(record, Mapping)
        for name in ("prepared_core_artifact_sha256", "graph_record_sha256")
    ):
        raise EmbeddingClusterAnalysisError("Source-core checksums are incomplete.")
    input_provenance = manifest.get("input_provenance")
    if not isinstance(input_provenance, Mapping) or (
        input_provenance.get("run_id") != EXPECTED_RUN_ID
        or int(input_provenance.get("model_seed", -1)) != EXPECTED_MODEL_SEED
        or input_provenance.get("campaign_id") != CAMPAIGN_ID
    ):
        raise EmbeddingClusterAnalysisError("Input provenance is incomplete.")

    configuration = manifest.get("clustering_configuration")
    if not isinstance(configuration, Mapping) or (
        configuration.get("run_id") != EXPECTED_RUN_ID
        or configuration.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
        or tuple(configuration.get("core_order", ())) != CORE_NUMBERS
        or configuration.get("joint_clustering") is not True
        or configuration.get("distance_metric") != "cosine"
        or int(configuration.get("n_neighbors", 0)) <= 0
        or int(configuration.get("pca_components", 0)) <= 0
        or float(configuration.get("leiden_resolution", 0.0)) <= 0.0
        or int(configuration.get("random_seed", -1)) < 0
    ):
        raise EmbeddingClusterAnalysisError("Clustering configuration is incomplete.")
    cluster_counts = manifest.get("cluster_counts")
    ranges = manifest.get("cluster_size_ranges")
    dominated = manifest.get("core_dominated_gt_90pct")
    if not all(isinstance(value, Mapping) for value in (cluster_counts, ranges, dominated)):
        raise EmbeddingClusterAnalysisError("Cluster summaries are incomplete.")
    for representation, prefix in zip(
        REPRESENTATIONS, ("I", "C"), strict=True
    ):
        if int(cluster_counts.get(representation, 0)) <= 0:
            raise EmbeddingClusterAnalysisError("Cluster count is invalid.")
        size_range = ranges.get(representation)
        if (
            not isinstance(size_range, list)
            or len(size_range) != 2
            or int(size_range[0]) <= 0
            or int(size_range[0]) > int(size_range[1])
            or int(size_range[1]) > EXPECTED_TOTAL_CELLS
        ):
            raise EmbeddingClusterAnalysisError("Cluster size range is invalid.")
        dominated_labels = dominated.get(representation)
        if not isinstance(dominated_labels, list) or any(
            not str(label).startswith(prefix) for label in dominated_labels
        ):
            raise EmbeddingClusterAnalysisError("Core-dominance flags are invalid.")

    delta = manifest.get("delta_h_l2")
    if not isinstance(delta, Mapping) or int(delta.get("count", -1)) != EXPECTED_TOTAL_CELLS:
        raise EmbeddingClusterAnalysisError("Delta-norm summary is incomplete.")
    numeric_delta = (
        "minimum",
        "maximum",
        "mean",
        "median",
        "standard_deviation",
        "p01",
        "p99",
    )
    if any(not math.isfinite(float(delta.get(name, math.nan))) for name in numeric_delta):
        raise EmbeddingClusterAnalysisError("Delta-norm statistics are non-finite.")
    limits = delta.get("plotting_limits")
    if not isinstance(limits, Mapping) or (
        not math.isfinite(float(limits.get("vmin_global_p01", math.nan)))
        or not math.isfinite(float(limits.get("vmax_global_p99", math.nan)))
        or float(limits.get("vmin_global_p01", math.nan))
        >= float(limits.get("vmax_global_p99", math.nan))
        or limits.get("shared_across_all_six_cores") is not True
        or limits.get("raw_values_clipped_in_saved_data") is not False
    ):
        raise EmbeddingClusterAnalysisError("Delta plotting limits are invalid.")

    extraction_path = root / "embeddings" / "extraction_manifest.json"
    clustering_path = root / "clustering" / "clustering_manifest.json"
    if (
        manifest.get("extraction_manifest_sha256") != sha256_file(extraction_path)
        or manifest.get("clustering_manifest_sha256") != sha256_file(clustering_path)
    ):
        raise EmbeddingClusterAnalysisError("Stage manifest checksums changed.")
    extraction = _read_json(extraction_path, label="extraction stage manifest")
    clustering = _read_json(clustering_path, label="clustering stage manifest")
    _verify_self_hash(extraction, label="extraction stage manifest")
    _verify_self_hash(clustering, label="clustering stage manifest")
    if (
        extraction.get("schema") != EXTRACTION_SCHEMA
        or extraction.get("status") != "complete"
        or extraction.get("run_id") != EXPECTED_RUN_ID
        or tuple(extraction.get("core_order", ())) != CORE_NUMBERS
        or int(extraction.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS
        or int(extraction.get("embedding_dimension", -1)) != 256
        or clustering.get("schema") != CLUSTERING_SCHEMA
        or clustering.get("status") != "complete"
        or clustering.get("run_id") != EXPECTED_RUN_ID
        or tuple(clustering.get("core_order", ())) != CORE_NUMBERS
        or int(clustering.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS
        or clustering.get("independent_representation_pipelines") is not True
        or clustering.get("configuration") != configuration
    ):
        raise EmbeddingClusterAnalysisError("Stage manifest semantics are invalid.")
    for stage in (extraction, clustering):
        stage_files = stage.get("files")
        if not isinstance(stage_files, Mapping) or any(
            files.get(relative) != record for relative, record in stage_files.items()
        ):
            raise EmbeddingClusterAnalysisError("Stage file receipts changed.")

    figure_files = {
        *(_required_combined_figures()),
        *(_required_per_core_figures()),
    }
    plotting = manifest.get("plotting")
    if not isinstance(plotting, Mapping) or (
        int(plotting.get("combined_figure_count", -1)) != 6
        or int(plotting.get("per_core_figure_count", -1)) != 36
        or plotting.get("one_dot_per_cell") is not True
        or plotting.get("point_layer_rasterized_in_pdf") is not True
        or not isinstance(plotting.get("files"), Mapping)
        or set(plotting["files"]) != figure_files
        or any(files.get(name) != record for name, record in plotting["files"].items())
    ):
        raise EmbeddingClusterAnalysisError("Plotting receipt is incomplete.")
    combined = manifest.get("combined_figure_checksums")
    if not isinstance(combined, Mapping) or (
        set(combined) != set(_required_combined_figures())
        or any(files.get(name) != record for name, record in combined.items())
    ):
        raise EmbeddingClusterAnalysisError("Combined figure checksums are incomplete.")
    interpretation = manifest.get("interpretation")
    if not isinstance(interpretation, Mapping) or any(
        interpretation.get(name) is not False
        for name in (
            "establishes_cell_type",
            "establishes_signaling",
            "establishes_biological_influence",
            "establishes_causality",
        )
    ) or interpretation.get("marker_and_pathology_validation_separate") is not True:
        raise EmbeddingClusterAnalysisError("Interpretation constraints are incomplete.")


def verify_embedding_cluster_analysis_bundle(
    bundle_path: str | Path,
) -> Mapping[str, Any]:
    root = Path(bundle_path).expanduser().resolve(strict=True)
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise EmbeddingClusterAnalysisError("Analysis bundle is missing or has symlinks.")
    manifest = _read_json(root / "manifest.json", label="final analysis manifest")
    _verify_self_hash(manifest, label="final analysis manifest")
    if (
        manifest.get("schema") != ANALYSIS_SCHEMA
        or manifest.get("status") != "complete"
        or tuple(manifest.get("core_order", ())) != CORE_NUMBERS
        or int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS
    ):
        raise EmbeddingClusterAnalysisError("Final analysis manifest identity is invalid.")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, Mapping) or dict(expected_files) != _file_manifest(root):
        raise EmbeddingClusterAnalysisError("Final analysis file checksums changed.")
    required = _required_analysis_files()
    missing = sorted(required - set(expected_files))
    if missing:
        raise EmbeddingClusterAnalysisError(
            "Final analysis lacks required files: " + ", ".join(missing)
        )
    _verify_final_semantics(root, manifest, expected_files)
    return manifest


def run_embedding_cluster_analysis(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None,
    checkpoint: str | Path | None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    device: str = "cuda:0",
    output_dir: str | Path | None = None,
) -> Mapping[str, Any]:
    """Run the complete resumable six-core embedding-cluster analysis."""

    if isinstance(n_neighbors, bool) or int(n_neighbors) <= 0:
        raise EmbeddingClusterAnalysisError("--n-neighbors must be positive.")
    if not math.isfinite(float(leiden_resolution)) or float(leiden_resolution) <= 0:
        raise EmbeddingClusterAnalysisError("--leiden-resolution must be positive.")
    if isinstance(pca_components, bool) or int(pca_components) <= 0:
        raise EmbeddingClusterAnalysisError("--pca-components must be positive.")
    if isinstance(random_seed, bool) or int(random_seed) < 0:
        raise EmbeddingClusterAnalysisError("--random-seed must be non-negative.")
    inputs = resolve_analysis_inputs(
        registry=registry,
        paths=paths,
        run_id=run_id,
        checkpoint=checkpoint,
    )
    output_namespace = (
        paths.report_root
        / "analyses"
        / "cancer_6core_embedding_clustering"
    ).resolve(strict=False)
    if output_dir is None:
        output_candidate = (
            paths.report_root
            / "analyses"
            / "cancer_6core_embedding_clustering"
            / inputs.run_id
        )
    else:
        output_candidate = Path(output_dir).expanduser()
        if not output_candidate.is_absolute():
            output_candidate = paths.project_root / output_candidate
    if output_candidate.is_symlink():
        raise EmbeddingClusterAnalysisError("Analysis output may not be a symlink.")
    output_root = output_candidate.resolve(strict=False)
    if (
        output_root == output_namespace
        or not output_root.is_relative_to(output_namespace)
    ):
        raise EmbeddingClusterAnalysisError(
            "Analysis output must be a run-specific directory beneath "
            f"{output_namespace}."
        )
    if output_root.exists() and not output_root.is_dir():
        raise EmbeddingClusterAnalysisError("Analysis output must be a directory.")
    if output_root.exists():
        if any(path.is_symlink() for path in output_root.rglob("*")):
            raise EmbeddingClusterAnalysisError(
                "Analysis output may not contain symbolic links."
            )
        allowed_top_level = {
            "embeddings",
            "clustering",
            "tables",
            "figures",
            "README.md",
            "manifest.json",
        }
        unexpected = sorted(
            path.name
            for path in output_root.iterdir()
            if path.name not in allowed_top_level
        )
        if unexpected:
            raise EmbeddingClusterAnalysisError(
                "Analysis output contains unrecognized entries: "
                + ", ".join(unexpected)
            )
    final_manifest_path = output_root / "manifest.json"
    if final_manifest_path.is_file():
        manifest = verify_embedding_cluster_analysis_bundle(output_root)
        requested_identity = {
            "run_id": inputs.run_id,
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "n_neighbors": int(n_neighbors),
            "distance_metric": "cosine",
            "leiden_resolution": float(leiden_resolution),
            "pca_components": int(pca_components),
            "random_seed": int(random_seed),
        }
        stored_configuration = manifest.get("clustering_configuration")
        if not isinstance(stored_configuration, Mapping) or any(
            stored_configuration.get(name) != value
            for name, value in requested_identity.items()
        ):
            raise EmbeddingClusterAnalysisError(
                "Completed output does not match the requested checkpoint or "
                "clustering parameters. Use a distinct --output-dir."
            )
        return {
            "status": "complete",
            "resumed": True,
            "output_dir": output_root.as_posix(),
            "manifest_sha256": sha256_file(final_manifest_path),
            "run_id": manifest["run_id"],
            "total_cells": manifest["total_cells"],
            "cluster_counts": manifest["cluster_counts"],
        }
    output_root.mkdir(parents=True, exist_ok=True)
    extraction = extract_intermediate_embeddings(
        inputs=inputs,
        output_root=output_root,
        device=device,
    )
    clustering = build_clustering_outputs(
        inputs=inputs,
        output_root=output_root,
        extraction_receipt=extraction,
        n_neighbors=int(n_neighbors),
        leiden_resolution=float(leiden_resolution),
        pca_components=int(pca_components),
        random_seed=int(random_seed),
    )
    plotting = render_spatial_outputs(
        output_root=output_root, clustering_receipt=clustering, dpi=300
    )
    readme_path = output_root / "README.md"
    _atomic_write_text(
        readme_path,
        _render_readme(
            inputs=inputs,
            extraction=extraction,
            clustering=clustering,
            output_root=output_root,
        ),
    )
    files = _file_manifest(output_root)
    combined_checksums = {
        relative: files[relative] for relative in _required_combined_figures()
    }
    manifest = _receipt_with_self_hash(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "analysis_classification": "exploratory_post_hoc_locked_readout",
            "campaign_id": CAMPAIGN_ID,
            "run_id": inputs.run_id,
            "model_seed": int(inputs.checkpoint_payload["model_seed"]),
            "checkpoint_path": inputs.checkpoint_path.as_posix(),
            "checkpoint_sha256": inputs.checkpoint_sha256,
            "training_source_commit": inputs.provenance["training_source"].get("commit"),
            "analysis_source_commit": inputs.provenance["analysis_source"].get("commit"),
            "model_construction": inputs.provenance["model_construction"],
            "preprocessing_checksum": inputs.provenance["cohort_manifest"].get(
                "preprocessing_statistics_sha256"
            ),
            "input_provenance": inputs.provenance,
            "source_core_artifacts": [
                {
                    "alias": core["alias"],
                    "core_number": core["core_number"],
                    "prepared_core_artifact_sha256": core[
                        "prepared_core_artifact_sha256"
                    ],
                    "graph_record_sha256": core["graph_record_sha256"],
                }
                for core in extraction["cores"]
            ],
            "core_order": list(CORE_NUMBERS),
            "cell_counts": {
                str(core["core_number"]): int(core["cell_count"])
                for core in extraction["cores"]
            },
            "total_cells": int(extraction["total_cells"]),
            "embedding_shapes": {
                str(core["core_number"]): {
                    "h0": core["array_shapes"]["h0"],
                    "hL": core["array_shapes"]["hL"],
                }
                for core in extraction["cores"]
            },
            "cluster_counts": clustering["cluster_counts"],
            "cluster_size_ranges": clustering["cluster_size_ranges"],
            "clustering_configuration": clustering["configuration"],
            "core_dominated_gt_90pct": clustering["core_dominated_gt_90pct"],
            "delta_h_l2": clustering["delta_h_l2"],
            "extraction_manifest_sha256": sha256_file(
                output_root / "embeddings" / "extraction_manifest.json"
            ),
            "clustering_manifest_sha256": sha256_file(
                output_root / "clustering" / "clustering_manifest.json"
            ),
            "plotting": plotting,
            "combined_figure_checksums": combined_checksums,
            "interpretation": {
                "intrinsic": "cell-own expression and metadata embedding patterns",
                "contextual": "patterns after graph-based neighborhood processing",
                "delta_h_l2": "magnitude of representation change after contextual processing",
                "establishes_cell_type": False,
                "establishes_signaling": False,
                "establishes_biological_influence": False,
                "establishes_causality": False,
                "marker_and_pathology_validation_separate": True,
            },
            "files": files,
        }
    )
    _atomic_write_json(final_manifest_path, manifest)
    verified = verify_embedding_cluster_analysis_bundle(output_root)
    return {
        "status": "complete",
        "resumed": False,
        "output_dir": output_root.as_posix(),
        "manifest_sha256": sha256_file(final_manifest_path),
        "run_id": verified["run_id"],
        "total_cells": verified["total_cells"],
        "cluster_counts": verified["cluster_counts"],
        "cluster_size_ranges": verified["cluster_size_ranges"],
        "core_dominated_gt_90pct": verified["core_dominated_gt_90pct"],
        "delta_h_l2": verified["delta_h_l2"],
        "combined_figures": [
            (output_root / relative).as_posix()
            for relative in _required_combined_figures()
        ],
    }


__all__ = [
    "ANALYSIS_SCHEMA",
    "CLUSTERING_SCHEMA",
    "CoreEmbeddings",
    "DEFAULT_LEIDEN_RESOLUTION",
    "DEFAULT_N_NEIGHBORS",
    "DEFAULT_PCA_COMPONENTS",
    "DEFAULT_RANDOM_SEED",
    "EmbeddingClusterAnalysisError",
    "KNNGraphResult",
    "PCAResult",
    "build_faiss_cosine_knn_graph",
    "build_cell_index_frame",
    "build_clustering_outputs",
    "cluster_summary_tables",
    "cluster_joint_representation",
    "concatenate_representation",
    "delta_norm_statistics",
    "deterministic_pca",
    "deterministic_glasbey_palette",
    "extract_intermediate_embeddings",
    "load_all_core_embeddings",
    "render_spatial_outputs",
    "requested_panel_order",
    "resolve_analysis_inputs",
    "run_embedding_cluster_analysis",
    "run_seeded_leiden",
    "spatial_plot_spec",
    "verify_embedding_cluster_analysis_bundle",
]

#!/usr/bin/env python3
"""Independently audit the locked adjacency-ablation preparation semantics.

This entry point deliberately does not call the campaign graph or standardizer
builders.  It reconstructs both from the manifest-bound upstream arrays and
writes a checksum-bound, write-once receipt.  It never reads model outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import load_yaml_mapping  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402


CAMPAIGN_ID = "cmp_20260802_adjacent_normal_grouped_adjacency_ablation"
RECEIPT_KIND = "adjacency_ablation_preparation_semantic_audit_v1"
CONTRACT_SHA256 = (
    "08e4040ce8b0a3535c7bef1cbf896c68bc5e11693cb13bbdfb3b992467742eff"
)
MATERIALIZATION_CHECKSUM = (
    "0cc606d73a4979808581a032138cc18ef629952ed1a4884903ecd229bdfc4303"
)
CONTRACT_REFERENCE = Path("experiments/campaigns") / CAMPAIGN_ID / (
    "frozen_task_contract.yaml"
)
MATERIALIZATION_REFERENCE = (
    Path("scratch/locked_campaigns") / CAMPAIGN_ID / "materialization_receipt.json"
)
PREPARED_REFERENCE = Path(
    "data/processed/adjacent_normal_grouped_adjacency_ablation_v1"
)
DEFAULT_OUTPUT = Path("scratch/locked_campaigns") / CAMPAIGN_ID / (
    "preparation_semantic_audit_receipt.json"
)
IMPLEMENTATION_REFERENCE = Path(
    "scripts/train/audit_adjacency_preparation_semantics.py"
)
CORE_ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))


class SemanticAuditError(RuntimeError):
    """Raised when a locked input or reconstructed semantic object differs."""


def _strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SemanticAuditError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                SemanticAuditError(f"non-finite JSON value in {path}: {value}")
            ),
            object_pairs_hook=unique_object,
        )
    except SemanticAuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SemanticAuditError(f"cannot read strict JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise SemanticAuditError(f"JSON root is not a mapping: {path}")
    return dict(value)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticAuditError(f"{label} is not a mapping")
    return value


def _project_path(project_root: Path, reference: object, label: str) -> Path:
    relative = Path(str(reference))
    if relative.is_absolute() or ".." in relative.parts:
        raise SemanticAuditError(f"unsafe {label} reference: {reference}")
    path = project_root / relative
    if path.is_symlink() or not path.exists():
        raise SemanticAuditError(f"missing or symlinked {label}: {reference}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ndarray_sha256(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise SemanticAuditError("object arrays are prohibited")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"bagm.ndarray.v1\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _joined_sha256(namespace: str, fields: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    for field in fields:
        encoded = str(field).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _derived_seed(base_seed: int, *parts: object) -> int:
    digest = _joined_sha256(
        "bagm.adjacency_ablation.seed.v1",
        [str(int(base_seed)), *(str(part) for part in parts)],
    )
    return int.from_bytes(bytes.fromhex(digest[:16]), "big", signed=False)


def _canonical_edges(edges: np.ndarray) -> np.ndarray:
    result = np.asarray(edges, dtype=np.int64)
    if result.ndim != 2 or result.shape[0] != 2:
        raise SemanticAuditError("reconstructed edges are not shaped [2, E]")
    if result.shape[1]:
        result = result[:, np.lexsort((result[1], result[0]))]
    return np.ascontiguousarray(result, dtype=np.int64)


def rebuild_fixed_adjacencies(
    coordinates_um: np.ndarray,
    raw_fov: np.ndarray,
    *,
    alias: str,
    k: int,
    radius_um: float,
    null_seed: int,
) -> dict[str, np.ndarray]:
    """Rebuild the three arms without importing the production graph builder."""

    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    raw_labels = np.asarray(raw_fov)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise SemanticAuditError("coordinates must have shape [cells, 2]")
    if raw_labels.shape != (len(coordinates),) or not np.isfinite(coordinates).all():
        raise SemanticAuditError("FOV labels or coordinates are invalid")
    observed = tuple(sorted(int(value) for value in np.unique(raw_labels)))
    lookup = {value: index for index, value in enumerate(observed)}
    fov_group = np.asarray(
        [lookup[int(value)] for value in raw_labels], dtype=np.int16
    )
    # Production canonicalizes group labels to strings before np.unique.  Keep
    # that ordering here because it also fixes the RNG stream across FOVs.
    canonical_fov = np.asarray(
        [str(value) for value in fov_group.tolist()], dtype="U128"
    )
    candidates: set[tuple[int, int]] = set()
    for label in np.unique(canonical_fov):
        global_indices = np.flatnonzero(canonical_fov == label)
        local_coordinates = coordinates[global_indices]
        neighborhoods = cKDTree(local_coordinates).query_ball_point(
            local_coordinates, r=float(radius_um)
        )
        for local_source, local_neighbors in enumerate(neighborhoods):
            source = int(global_indices[local_source])
            ranked: list[tuple[float, int]] = []
            for local_target in local_neighbors:
                target = int(global_indices[int(local_target)])
                if target == source:
                    continue
                distance = float(
                    np.linalg.norm(coordinates[target] - coordinates[source])
                )
                if 0.0 <= distance <= float(radius_um):
                    ranked.append((distance, target))
            ranked.sort(key=lambda item: (item[0], item[1]))
            candidates.update((source, target) for _, target in ranked[: int(k)])
    undirected = sorted(
        {(min(source, target), max(source, target)) for source, target in candidates}
    )
    directed_pairs = undirected + [(target, source) for source, target in undirected]
    off_diagonal = _canonical_edges(
        np.asarray(directed_pairs, dtype=np.int64).reshape(-1, 2).T
    )
    nodes = np.arange(len(coordinates), dtype=np.int64)
    loops = np.stack([nodes, nodes], axis=0)
    spatial = _canonical_edges(np.concatenate([off_diagonal, loops], axis=1))
    identity = loops
    generator = np.random.default_rng(
        _derived_seed(null_seed, "position_null", alias)
    )
    position_assignment = nodes.copy()
    for label in np.unique(canonical_fov):
        positions = np.flatnonzero(canonical_fov == label)
        position_assignment[positions] = generator.permutation(positions)
    position_null = _canonical_edges(
        np.concatenate([position_assignment[off_diagonal], loops], axis=1)
    )
    return {
        "fov_group": fov_group,
        "off_diagonal": off_diagonal,
        "spatial": spatial,
        "identity": identity,
        "position_assignment": position_assignment,
        "position_permuted_null": position_null,
    }


def equal_core_standardizer(
    moments_by_alias: Mapping[str, tuple[np.ndarray, np.ndarray]],
    train_aliases: Sequence[str],
    *,
    scale_floor: float,
) -> dict[str, Any]:
    """Assemble equal-core moments from independently computed per-core moments."""

    aliases = tuple(str(alias) for alias in train_aliases)
    mean = np.stack([moments_by_alias[alias][0] for alias in aliases]).mean(axis=0)
    second = np.stack([moments_by_alias[alias][1] for alias in aliases]).mean(
        axis=0
    )
    variance = np.maximum(second - np.square(mean), 0.0)
    scale = np.maximum(np.sqrt(variance), float(scale_floor))
    checksum = _joined_sha256(
        "bagm.equal_core_log1p_standardizer.v1",
        [
            *aliases,
            repr(float(scale_floor)),
            ndarray_sha256(mean),
            ndarray_sha256(variance),
            ndarray_sha256(scale),
        ],
    )
    return {
        "mean": mean,
        "variance": variance,
        "scale": scale,
        "checksum": checksum,
    }


def _verify_signed(payload: Mapping[str, Any], label: str) -> str:
    checksum = str(payload.get("checksum", ""))
    unsigned = dict(payload)
    unsigned.pop("checksum", None)
    if len(checksum) != 64 or canonical_sha256(unsigned) != checksum:
        raise SemanticAuditError(f"{label} checksum does not verify")
    return checksum


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticAuditError(message)


def audit_preparation(
    *,
    project_root: Path,
    contract_reference: Path = CONTRACT_REFERENCE,
    materialization_reference: Path = MATERIALIZATION_REFERENCE,
    prepared_reference: Path = PREPARED_REFERENCE,
) -> dict[str, Any]:
    """Run the locked semantic reconstruction and return an unsigned receipt."""

    started = time.perf_counter()
    project_root = project_root.resolve()
    contract_path = _project_path(project_root, contract_reference, "contract")
    contract = load_yaml_mapping(contract_path)
    _require(_sha256_file(contract_path) == CONTRACT_SHA256, "contract changed")
    _require(contract.get("campaign_id") == CAMPAIGN_ID, "contract campaign changed")
    graph_contract = _mapping(contract.get("graph"), "contract graph")
    preprocessing_contract = _mapping(
        contract.get("preprocessing"), "contract preprocessing"
    )

    materialization_path = _project_path(
        project_root, materialization_reference, "materialization receipt"
    )
    materialization = _strict_json(materialization_path)
    materialization_checksum = _verify_signed(
        materialization, "materialization receipt"
    )
    _require(
        materialization_checksum == MATERIALIZATION_CHECKSUM,
        "materialization identity changed",
    )
    _require(
        materialization.get("campaign_id") == CAMPAIGN_ID,
        "materialization campaign changed",
    )
    materialized_contract = _mapping(
        materialization.get("frozen_contract"), "materialized contract"
    )
    _require(
        materialized_contract.get("reference") == contract_reference.as_posix()
        and materialized_contract.get("sha256") == CONTRACT_SHA256,
        "materialization contract binding changed",
    )

    prepared_root = _project_path(project_root, prepared_reference, "prepared root")
    prepared_manifest_path = prepared_root / "manifest.json"
    prepared_manifest = _strict_json(prepared_manifest_path)
    materialized_prepared = _mapping(
        materialization.get("prepared_manifest"), "materialized prepared manifest"
    )
    prepared_file_sha = _sha256_file(prepared_manifest_path)
    _require(
        materialized_prepared.get("reference")
        == (prepared_reference / "manifest.json").as_posix()
        and materialized_prepared.get("file_sha256") == prepared_file_sha,
        "materialized prepared manifest binding changed",
    )
    content = {
        key: value
        for key, value in prepared_manifest.items()
        if key not in {"artifact_id", "content_sha256"}
    }
    content_sha = canonical_sha256(content)
    _require(
        prepared_manifest.get("content_sha256") == content_sha
        and prepared_manifest.get("artifact_id") == content_sha[:16],
        "prepared manifest content identity changed",
    )
    _require(
        materialized_prepared.get("content_sha256") == content_sha,
        "materialization prepared content binding changed",
    )
    cores = _mapping(prepared_manifest.get("cores"), "prepared cores")
    folds = _mapping(prepared_manifest.get("folds"), "prepared folds")
    _require(set(cores) == set(CORE_ALIASES), "prepared core inventory changed")
    _require(set(folds) == {str(index) for index in range(5)}, "fold inventory changed")
    provenance = {
        str(row.get("core_alias")): row
        for row in prepared_manifest.get("input_provenance", [])
        if isinstance(row, Mapping) and row.get("core_alias")
    }
    _require(set(provenance) == set(CORE_ALIASES), "upstream provenance changed")
    files = _mapping(prepared_manifest.get("files"), "prepared file checksums")

    moments_by_alias: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    core_audits: list[dict[str, Any]] = []
    for alias in CORE_ALIASES:
        record = _mapping(cores[alias], f"prepared core {alias}")
        upstream_record = _mapping(provenance[alias], f"upstream {alias}")
        upstream_root = _project_path(
            project_root, upstream_record.get("reference"), f"upstream {alias}"
        )
        upstream_manifest_path = upstream_root / "manifest.json"
        upstream_data_path = upstream_root / "prepared_data.npz"
        _require(
            _sha256_file(upstream_manifest_path)
            == upstream_record.get("manifest_sha256"),
            f"{alias} upstream manifest changed",
        )
        _require(
            _sha256_file(upstream_data_path)
            == upstream_record.get("prepared_data_sha256"),
            f"{alias} upstream data changed",
        )
        with np.load(upstream_data_path, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates_um"], dtype=np.float64)
            raw_fov = np.asarray(archive["fov"])
            counts = np.asarray(archive["expression_counts"])
            qc_passed = np.asarray(archive["qc_passed"], dtype=np.bool_)
        rebuilt = rebuild_fixed_adjacencies(
            coordinates,
            raw_fov,
            alias=alias,
            k=int(graph_contract["k"]),
            radius_um=float(graph_contract["radius_um"]),
            null_seed=int(graph_contract["null_seed"]),
        )
        data_reference = str(record["data_reference"])
        data_path = _project_path(
            project_root, prepared_reference / data_reference, f"prepared data {alias}"
        )
        _require(_sha256_file(data_path) == files[data_reference], f"{alias} data file changed")
        with np.load(data_path, allow_pickle=False) as archive:
            data_exact = all(
                (
                    np.array_equal(archive["expression_counts"], counts),
                    np.array_equal(archive["coordinates_um"], coordinates),
                    np.array_equal(archive["fov_group"], rebuilt["fov_group"]),
                    np.array_equal(archive["qc_passed"], qc_passed),
                )
            )
        adjacency_reference = str(record["adjacency_reference"])
        adjacency_path = _project_path(
            project_root,
            prepared_reference / adjacency_reference,
            f"prepared adjacency {alias}",
        )
        _require(
            _sha256_file(adjacency_path) == files[adjacency_reference],
            f"{alias} adjacency file changed",
        )
        with np.load(adjacency_path, allow_pickle=False) as archive:
            exact = {
                "spatial": np.array_equal(
                    archive["spatial_edge_index"], rebuilt["spatial"]
                ),
                "identity": np.array_equal(
                    archive["isolated_edge_index"], rebuilt["identity"]
                ),
                "position_permuted_null": np.array_equal(
                    archive["position_permuted_null_edge_index"],
                    rebuilt["position_permuted_null"],
                ),
            }
        graph_metadata = _mapping(record.get("graph_metadata"), f"graph {alias}")
        arms = _mapping(graph_metadata.get("arms"), f"graph arms {alias}")
        reconstructed_checksums = {
            "spatial": ndarray_sha256(rebuilt["spatial"]),
            "identity": ndarray_sha256(rebuilt["identity"]),
            "position_permuted_null": ndarray_sha256(
                rebuilt["position_permuted_null"]
            ),
            "off_diagonal": ndarray_sha256(rebuilt["off_diagonal"]),
            "position_assignment": ndarray_sha256(rebuilt["position_assignment"]),
        }
        checksum_exact = all(
            (
                reconstructed_checksums["spatial"]
                == _mapping(arms["spatial"], "spatial arm")["checksum"],
                reconstructed_checksums["identity"]
                == _mapping(arms["isolated"], "isolated arm")["checksum"],
                reconstructed_checksums["position_permuted_null"]
                == _mapping(arms["position_permuted_null"], "null arm")[
                    "checksum"
                ],
                reconstructed_checksums["position_assignment"]
                == graph_metadata["position_assignment_checksum"],
            )
        )
        bundle_checksum = _joined_sha256(
            "bagm.adjacency_bundle.v1",
            [
                reconstructed_checksums["spatial"],
                reconstructed_checksums["identity"],
                reconstructed_checksums["position_permuted_null"],
                reconstructed_checksums["off_diagonal"],
                reconstructed_checksums["position_assignment"],
            ],
        )
        bundle_exact = (
            bundle_checksum
            == graph_metadata.get("checksum")
            == record.get("graph_bundle_checksum")
        )
        _require(
            data_exact and all(exact.values()) and checksum_exact and bundle_exact,
            f"{alias} semantic graph reconstruction differs",
        )
        transformed = np.log1p(counts.astype(np.float64, copy=False))
        moments_by_alias[alias] = (
            transformed.mean(axis=0, dtype=np.float64),
            np.square(transformed).mean(axis=0, dtype=np.float64),
        )
        core_audits.append(
            {
                "core_alias": alias,
                "upstream": {
                    "reference": upstream_record["reference"],
                    "manifest_sha256": upstream_record["manifest_sha256"],
                    "prepared_data_sha256": upstream_record["prepared_data_sha256"],
                },
                "n_cells": int(len(coordinates)),
                "n_fov_groups": int(np.unique(rebuilt["fov_group"]).size),
                "downstream_data_exact": data_exact,
                "exact_matches": exact,
                "checksum_matches": {
                    "all_arm_and_position_checksums": checksum_exact,
                    "graph_bundle": bundle_exact,
                },
                "reconstructed_checksums": reconstructed_checksums,
                "graph_bundle_checksum": bundle_checksum,
            }
        )

    fold_audits: list[dict[str, Any]] = []
    for fold_index in range(5):
        record = _mapping(folds[str(fold_index)], f"fold {fold_index}")
        train_aliases = tuple(str(value) for value in record["train_aliases"])
        _require(len(train_aliases) == 7, f"fold {fold_index} train size changed")
        rebuilt = equal_core_standardizer(
            moments_by_alias,
            train_aliases,
            scale_floor=float(preprocessing_contract["scale_floor"]),
        )
        reference = str(record["preprocessing_reference"])
        path = _project_path(
            project_root, prepared_reference / reference, f"fold {fold_index} preprocessing"
        )
        file_exact = (
            _sha256_file(path)
            == files[reference]
            == record["preprocessing_file_sha256"]
        )
        with np.load(path, allow_pickle=False) as archive:
            mean_exact = np.array_equal(archive["log1p_mean"], rebuilt["mean"])
            scale_exact = np.array_equal(archive["log1p_scale"], rebuilt["scale"])
        mean_sha = ndarray_sha256(rebuilt["mean"])
        scale_sha = ndarray_sha256(rebuilt["scale"])
        array_checksums = _mapping(
            record["preprocessing_arrays"], f"fold {fold_index} arrays"
        )
        checksum_exact = all(
            (
                mean_sha == _mapping(array_checksums["log1p_mean"], "mean")["sha256"],
                scale_sha
                == _mapping(array_checksums["log1p_scale"], "scale")["sha256"],
                rebuilt["checksum"] == record["standardizer_checksum"],
            )
        )
        _require(
            file_exact and mean_exact and scale_exact and checksum_exact,
            f"fold {fold_index} equal-core preprocessing differs",
        )
        fold_audits.append(
            {
                "fold": fold_index,
                "train_aliases": list(train_aliases),
                "exact_matches": {
                    "mean": mean_exact,
                    "scale": scale_exact,
                    "array_and_standardizer_checksums": checksum_exact,
                    "preprocessing_file_checksum": file_exact,
                },
                "reconstructed_checksums": {
                    "mean": mean_sha,
                    "variance": ndarray_sha256(rebuilt["variance"]),
                    "scale": scale_sha,
                    "standardizer": rebuilt["checksum"],
                },
            }
        )

    selection_rows = [
        row
        for row in prepared_manifest.get("input_provenance", [])
        if isinstance(row, Mapping) and row.get("role") == "protected_ten_core_selection"
    ]
    _require(len(selection_rows) == 1, "protected selection binding changed")
    selection = selection_rows[0]
    selection_path = _project_path(
        project_root, selection.get("reference"), "protected selection"
    )
    _require(
        _sha256_file(selection_path) == selection.get("sha256"),
        "protected selection checksum changed",
    )
    return {
        "schema_version": 1,
        "receipt_kind": RECEIPT_KIND,
        "campaign_id": CAMPAIGN_ID,
        "scientific_contract": {
            "reference": contract_reference.as_posix(),
            "sha256": CONTRACT_SHA256,
        },
        "materialization": {
            "reference": materialization_reference.as_posix(),
            "file_sha256": _sha256_file(materialization_path),
            "receipt_checksum": materialization_checksum,
        },
        "prepared_artifact": {
            "reference": prepared_reference.as_posix(),
            "manifest_file_sha256": prepared_file_sha,
            "content_sha256": content_sha,
            "artifact_id": prepared_manifest["artifact_id"],
            "checksums_file_sha256": _sha256_file(
                prepared_root / "checksums.sha256"
            ),
        },
        "protected_selection": {
            "reference": selection["reference"],
            "sha256": selection["sha256"],
        },
        "implementation": {
            "reference": IMPLEMENTATION_REFERENCE.as_posix(),
            "file_sha256": _sha256_file(project_root / IMPLEMENTATION_REFERENCE),
            "independent_of_production_graph_and_standardizer_builders": True,
        },
        "audit_scope": {
            "performance_outcomes_read": False,
            "core_count": len(core_audits),
            "fold_count": len(fold_audits),
        },
        "cores": core_audits,
        "folds": fold_audits,
        "all_checks_passed": True,
        "runtime": {"wall_seconds": round(time.perf_counter() - started, 6)},
    }


def signed_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["checksum"] = canonical_sha256(result)
    return result


def write_once(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        dict(receipt), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise SemanticAuditError(f"refusing to overwrite audit receipt: {path}") from error
    path.chmod(0o444)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_preparation(project_root=args.project_root)
    receipt = signed_receipt(payload)
    if not args.no_write:
        write_once(args.output.resolve(), receipt)
    print(
        json.dumps(
            {
                "all_checks_passed": receipt["all_checks_passed"],
                "checksum": receipt["checksum"],
                "cores": len(receipt["cores"]),
                "folds": len(receipt["folds"]),
                "output": None if args.no_write else str(args.output.resolve()),
                "wall_seconds": receipt["runtime"]["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

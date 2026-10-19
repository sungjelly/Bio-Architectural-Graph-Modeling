"""Validated ranking helpers for gene-to-gene sensitivity matrices.

The functions in this module deliberately distinguish a Jacobian entry from a
correlation coefficient.  ``rank_mutual_sensitivity_pairs`` ranks the absolute
off-diagonal derivative magnitude used by the upstream GeneMAE audit, whereas
``profile_spearman_matrices`` compares complete signed sensitivity profiles.
Neither statistic is a biological or causal effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


class JacobianPairAnalysisError(ValueError):
    """Raised when a Jacobian artifact or requested ranking is invalid."""


@dataclass(frozen=True, slots=True)
class JacobianArtifact:
    """Validated upstream gene-to-gene sensitivity artifact."""

    genes: tuple[str, ...]
    signed_directed: np.ndarray
    symmetric_absolute: np.ndarray
    sha256: str
    path: Path
    adaptation_note: str
    published_transform: str


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest for *path* without loading it all at once."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_mapping(
    payload: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise JacobianPairAnalysisError(f"{key!r} must be a mapping")
    return value


def load_jacobian_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_gene_count: int = 39,
    absolute_tolerance: float = 1e-12,
) -> JacobianArtifact:
    """Load and fail-closed validate the finalized GeneMAE matrix artifact."""

    artifact_path = Path(path).resolve(strict=True)
    observed_sha256 = file_sha256(artifact_path)
    if observed_sha256 != expected_sha256:
        raise JacobianPairAnalysisError(
            "Jacobian artifact checksum mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    with artifact_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise JacobianPairAnalysisError("Jacobian report must contain a mapping")
    source = _nested_mapping(payload, "source_style_reproduction")

    raw_genes = source.get("marker_gene_order")
    if not isinstance(raw_genes, list) or any(
        not isinstance(gene, str) or not gene.strip() for gene in raw_genes
    ):
        raise JacobianPairAnalysisError(
            "marker_gene_order must be a list of non-empty strings"
        )
    genes = tuple(raw_genes)
    if len(genes) != expected_gene_count:
        raise JacobianPairAnalysisError(
            f"Expected {expected_gene_count} genes, observed {len(genes)}"
        )
    if len(set(genes)) != len(genes):
        raise JacobianPairAnalysisError("marker_gene_order contains duplicates")

    shape = (expected_gene_count, expected_gene_count)
    signed = np.asarray(
        source.get("equal_core_seven_seed_signed_directed_matrix"),
        dtype=np.float64,
    )
    symmetric = np.asarray(
        source.get(
            "equal_core_seven_seed_published_symmetric_absolute_matrix"
        ),
        dtype=np.float64,
    )
    if signed.shape != shape:
        raise JacobianPairAnalysisError(
            f"Signed matrix has shape {signed.shape}; expected {shape}"
        )
    if symmetric.shape != shape:
        raise JacobianPairAnalysisError(
            f"Symmetric matrix has shape {symmetric.shape}; expected {shape}"
        )
    if not bool(np.isfinite(signed).all()):
        raise JacobianPairAnalysisError("Signed matrix contains nonfinite values")
    if not bool(np.isfinite(symmetric).all()):
        raise JacobianPairAnalysisError(
            "Symmetric matrix contains nonfinite values"
        )
    expected_symmetric = 0.5 * (np.abs(signed) + np.abs(signed.T))
    if not np.allclose(
        symmetric,
        expected_symmetric,
        rtol=0.0,
        atol=float(absolute_tolerance),
    ):
        maximum_error = float(np.max(np.abs(symmetric - expected_symmetric)))
        raise JacobianPairAnalysisError(
            "Stored symmetric matrix does not reproduce "
            f"0.5*(abs(J)+abs(J.T)); max error={maximum_error}"
        )
    if not np.allclose(
        symmetric,
        symmetric.T,
        rtol=0.0,
        atol=float(absolute_tolerance),
    ):
        raise JacobianPairAnalysisError("Stored symmetric matrix is not symmetric")
    if bool((symmetric < 0).any()):
        raise JacobianPairAnalysisError(
            "Stored symmetric absolute matrix contains negative values"
        )

    signed = signed.copy()
    symmetric = symmetric.copy()
    signed.setflags(write=False)
    symmetric.setflags(write=False)
    return JacobianArtifact(
        genes=genes,
        signed_directed=signed,
        symmetric_absolute=symmetric,
        sha256=observed_sha256,
        path=artifact_path,
        adaptation_note=str(source.get("adaptation_note", "")),
        published_transform=str(source.get("published_transform", "")),
    )


def _validate_matrix_and_genes(
    genes: Sequence[str], matrix: np.ndarray, *, label: str
) -> tuple[tuple[str, ...], np.ndarray]:
    names = tuple(genes)
    if len(names) < 3 or len(set(names)) != len(names):
        raise JacobianPairAnalysisError(
            "genes must contain at least three unique names"
        )
    values = np.asarray(matrix, dtype=np.float64)
    expected = (len(names), len(names))
    if values.shape != expected:
        raise JacobianPairAnalysisError(
            f"{label} has shape {values.shape}; expected {expected}"
        )
    if not bool(np.isfinite(values).all()):
        raise JacobianPairAnalysisError(f"{label} contains nonfinite values")
    return names, values


def rank_mutual_sensitivity_pairs(
    genes: Sequence[str],
    signed_directed: np.ndarray,
    symmetric_absolute: np.ndarray,
    *,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Rank unique off-diagonal pairs by the upstream mutual magnitude."""

    names, signed = _validate_matrix_and_genes(
        genes, signed_directed, label="signed_directed"
    )
    _, symmetric = _validate_matrix_and_genes(
        names, symmetric_absolute, label="symmetric_absolute"
    )
    if top_n < 1:
        raise JacobianPairAnalysisError("top_n must be positive")
    pair_count = len(names) * (len(names) - 1) // 2
    if top_n > pair_count:
        raise JacobianPairAnalysisError(
            f"top_n={top_n} exceeds the {pair_count} unique off-diagonal pairs"
        )

    indexed: list[tuple[float, int, int]] = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            indexed.append((float(symmetric[first, second]), first, second))
    indexed.sort(key=lambda item: (-item[0], item[1], item[2]))

    rows: list[dict[str, Any]] = []
    for rank, (score, first, second) in enumerate(indexed[:top_n], start=1):
        first_from_second = float(signed[first, second])
        second_from_first = float(signed[second, first])
        dominant = (
            f"{names[first]}<-{names[second]}"
            if abs(first_from_second) >= abs(second_from_first)
            else f"{names[second]}<-{names[first]}"
        )
        rows.append(
            {
                "rank": rank,
                "gene_a": names[first],
                "gene_b": names[second],
                "mutual_absolute_sensitivity": score,
                "j_gene_a_target_gene_b_source": first_from_second,
                "j_gene_b_target_gene_a_source": second_from_first,
                "dominant_absolute_direction": dominant,
            }
        )
    return rows


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = rankdata(first, method="average")
    second_ranks = rankdata(second, method="average")
    first_centered = first_ranks - np.mean(first_ranks)
    second_centered = second_ranks - np.mean(second_ranks)
    denominator = float(
        np.sqrt(
            np.sum(first_centered * first_centered)
            * np.sum(second_centered * second_centered)
        )
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(first_centered * second_centered) / denominator)


def profile_spearman_matrices(
    genes: Sequence[str], signed_directed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return receiver-row and source-column Spearman profile matrices.

    For each pair ``(i, j)``, coordinates ``i`` and ``j`` are omitted before
    correlation.  This prevents either gene's diagonal/self entry or their
    direct pair entries from mechanically dominating profile similarity.
    """

    names, signed = _validate_matrix_and_genes(
        genes, signed_directed, label="signed_directed"
    )
    receiver = np.eye(len(names), dtype=np.float64)
    source = np.eye(len(names), dtype=np.float64)
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            keep = np.ones(len(names), dtype=bool)
            keep[[first, second]] = False
            receiver_value = _spearman(
                signed[first, keep], signed[second, keep]
            )
            source_value = _spearman(
                signed[keep, first], signed[keep, second]
            )
            receiver[first, second] = receiver[second, first] = receiver_value
            source[first, second] = source[second, first] = source_value
    receiver.setflags(write=False)
    source.setflags(write=False)
    return receiver, source


def rank_profile_pairs(
    genes: Sequence[str],
    correlations: np.ndarray,
    *,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Rank unique gene pairs by descending finite profile correlation."""

    names, values = _validate_matrix_and_genes(
        genes, correlations, label="correlations"
    )
    if not np.allclose(values, values.T, rtol=0.0, atol=1e-12, equal_nan=True):
        raise JacobianPairAnalysisError("correlations must be symmetric")
    if top_n < 1:
        raise JacobianPairAnalysisError("top_n must be positive")
    indexed: list[tuple[float, int, int]] = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            value = float(values[first, second])
            if np.isfinite(value):
                indexed.append((value, first, second))
    if top_n > len(indexed):
        raise JacobianPairAnalysisError(
            f"top_n={top_n} exceeds the {len(indexed)} finite pairs"
        )
    indexed.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {
            "rank": rank,
            "gene_a": names[first],
            "gene_b": names[second],
            "spearman_rho": value,
        }
        for rank, (value, first, second) in enumerate(
            indexed[:top_n], start=1
        )
    ]


__all__ = [
    "JacobianArtifact",
    "JacobianPairAnalysisError",
    "file_sha256",
    "load_jacobian_artifact",
    "profile_spearman_matrices",
    "rank_mutual_sensitivity_pairs",
    "rank_profile_pairs",
]

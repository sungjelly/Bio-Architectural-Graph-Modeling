from __future__ import annotations

import json

import numpy as np
import pytest

from spatial_benchmark.jacobian_pair_analysis import (
    JacobianPairAnalysisError,
    file_sha256,
    load_jacobian_artifact,
    profile_spearman_matrices,
    rank_mutual_sensitivity_pairs,
    rank_profile_pairs,
)


def test_rank_mutual_sensitivity_retains_both_directions() -> None:
    genes = ("A", "B", "C")
    signed = np.asarray(
        [
            [4.0, -3.0, 0.5],
            [1.0, 5.0, -2.0],
            [0.25, 0.0, 6.0],
        ]
    )
    symmetric = 0.5 * (np.abs(signed) + np.abs(signed.T))

    rows = rank_mutual_sensitivity_pairs(
        genes, signed, symmetric, top_n=2
    )

    assert rows[0] == {
        "rank": 1,
        "gene_a": "A",
        "gene_b": "B",
        "mutual_absolute_sensitivity": 2.0,
        "j_gene_a_target_gene_b_source": -3.0,
        "j_gene_b_target_gene_a_source": 1.0,
        "dominant_absolute_direction": "A<-B",
    }
    assert rows[1]["gene_a"] == "B"
    assert rows[1]["gene_b"] == "C"


def test_profile_correlations_exclude_pair_and_self_coordinates() -> None:
    genes = ("A", "B", "C", "D", "E")
    signed = np.asarray(
        [
            [100.0, -100.0, 1.0, 2.0, 3.0],
            [-200.0, 200.0, 2.0, 4.0, 6.0],
            [1.0, 2.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 3.0, 0.0, 6.0],
            [3.0, 6.0, 4.0, 5.0, 0.0],
        ]
    )

    receiver, source = profile_spearman_matrices(genes, signed)

    assert receiver[0, 1] == pytest.approx(1.0)
    assert source[0, 1] == pytest.approx(1.0)
    assert np.allclose(np.diag(receiver), 1.0)
    assert np.allclose(np.diag(source), 1.0)
    ranked = rank_profile_pairs(genes, receiver, top_n=1)
    assert ranked[0]["spearman_rho"] == pytest.approx(1.0)


def test_load_artifact_verifies_checksum_shape_and_transform(tmp_path) -> None:
    signed = [[1.0, -2.0, 0.0], [4.0, 2.0, 1.0], [0.5, -3.0, 3.0]]
    values = np.asarray(signed)
    symmetric = (0.5 * (np.abs(values) + np.abs(values.T))).tolist()
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "source_style_reproduction": {
                    "marker_gene_order": ["A", "B", "C"],
                    "equal_core_seven_seed_signed_directed_matrix": signed,
                    "equal_core_seven_seed_published_symmetric_absolute_matrix": symmetric,
                    "adaptation_note": "test",
                    "published_transform": "0.5 * (abs(J) + abs(J.T))",
                }
            }
        ),
        encoding="utf-8",
    )

    artifact = load_jacobian_artifact(
        path,
        expected_sha256=file_sha256(path),
        expected_gene_count=3,
    )

    assert artifact.genes == ("A", "B", "C")
    assert artifact.signed_directed.flags.writeable is False
    assert artifact.symmetric_absolute.flags.writeable is False


def test_load_artifact_rejects_checksum_mismatch(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        JacobianPairAnalysisError, match="checksum mismatch"
    ):
        load_jacobian_artifact(
            path,
            expected_sha256="0" * 64,
            expected_gene_count=3,
        )

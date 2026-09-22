from __future__ import annotations

import numpy as np
import pytest

from spatial_benchmark.cli import build_parser
from spatial_benchmark.paths import current_paths
import spatial_benchmark.so2_hidden_layer_progression as progression


def test_progression_cli_defaults_match_verified_hl_pipeline() -> None:
    arguments = build_parser().parse_args(
        ["analyze-so2-hidden-layer-progression"]
    )

    assert arguments.device == "cpu"
    assert arguments.n_neighbors == 30
    assert arguments.leiden_resolution == 1.0
    assert arguments.pca_components == 50
    assert arguments.random_seed == 20260825
    assert arguments.cpu_threads == 40
    assert arguments.cluster_workers == 4
    assert arguments.dpi == 300


def test_clustering_configuration_is_sparse_joint_and_cpu_only() -> None:
    configuration = progression.clustering_configuration(
        n_neighbors=30,
        leiden_resolution=1.0,
        pca_components=50,
        random_seed=20260825,
    )

    assert configuration["joint_cell_count"] == 246_063
    assert tuple(configuration["joint_core_order"]) == tuple(range(15, 29))
    assert configuration["mean_center"] is True
    assert configuration["l2_normalize_after_pca"] is True
    assert configuration["dense_cell_by_cell_matrix_constructed"] is False
    assert configuration["device"] == "cpu"

    with pytest.raises(progression.SO2HiddenLayerProgressionError, match="positive"):
        progression.clustering_configuration(
            n_neighbors=0,
            leiden_resolution=1.0,
            pca_components=50,
            random_seed=20260825,
        )


def test_partition_agreement_covers_all_pairs_and_qualifies_adjacency() -> None:
    size = progression.EXPECTED_TOTAL_CELLS
    base = np.arange(size, dtype=np.int64) % 4
    split = np.arange(size, dtype=np.int64) % 8
    labels = {
        "h0": base,
        "h1": base.copy(),
        "h2": split,
        "h3": split.copy(),
        "hL": base.copy(),
    }

    agreement = progression.partition_agreement_table(labels)

    assert len(agreement) == 10
    assert agreement["adjacent"].sum() == 4
    identical = agreement.loc[
        (agreement["source_layer"] == "h0")
        & (agreement["target_layer"] == "h1")
    ].iloc[0]
    assert identical["adjusted_rand_index"] == pytest.approx(1.0)
    assert identical["normalized_mutual_information"] == pytest.approx(1.0)
    assert identical["cell_count"] == size


def test_cluster_worker_count_is_bounded_before_starting_processes(tmp_path) -> None:
    with pytest.raises(
        progression.SO2HiddenLayerProgressionError, match="cluster_workers"
    ):
        progression.cluster_hidden_layers(
            output_root=tmp_path,
            extraction_root=tmp_path,
            run_id="synthetic",
            extraction_manifest_sha256="f" * 64,
            configuration={},
            workers=5,
        )


def test_source_snapshots_are_byte_identical_and_publication_is_no_replace(
    tmp_path,
) -> None:
    output = tmp_path / "staging"
    records = progression._source_code_records(current_paths().project_root, output)

    assert len(records) == 5
    for record in records.values():
        assert record["working_tree_file"] == record["snapshot_file"]
        assert (output / record["snapshot_path"]).is_file()

    destination = tmp_path / "published"
    progression._atomic_publish_directory_no_replace(output, destination)
    assert destination.is_dir()
    assert not output.exists()

    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(FileExistsError, match="replace"):
        progression._atomic_publish_directory_no_replace(second, destination)
    assert second.is_dir()

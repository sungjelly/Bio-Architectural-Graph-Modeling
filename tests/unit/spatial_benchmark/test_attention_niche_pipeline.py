from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import spatial_benchmark.attention_niche_pipeline as pipeline
from spatial_benchmark.attention_niche_pipeline import (
    CoreJobSpec,
    _combine_parquet_files,
    _incident_median,
    _integer_partition,
    _portable_core_receipts,
    _portable_file_receipts,
    _required_analysis_free_disk_gib,
    _remove_verified_core_work_tree,
    _stream_one_view,
    _validate_interpretation_identifier_minimization,
    _write_directed_core_table,
    _write_mutual_core_table,
)
from spatial_benchmark.attention_routing_niches import (
    retain_top_k_mutual_edges,
    summarize_mutual_routing_consensus,
)


def test_core_job_defaults_lock_primary_parameters(tmp_path: Path) -> None:
    spec = CoreJobSpec(
        alias="CAN-01",
        core_number=1,
        local_device=0,
        output_dir=tmp_path / "out",
        cohort_dir=tmp_path / "cohort",
        graph_dir=tmp_path / "graph",
        raw_dir=tmp_path / "raw",
        core_map_path=tmp_path / "core-map.csv",
        reconciliation_path=tmp_path / "policy.yaml",
        members=(),
    )

    assert spec.mask_views == 10
    assert spec.primary_top_k == 8
    assert spec.primary_resolution == 1.0
    assert spec.score_threshold == 1.0
    assert spec.support_threshold == 0.60
    assert spec.amp is False


def test_interpretation_assignments_reject_source_identifiers() -> None:
    permitted = pd.DataFrame(
        {
            "core_alias": ["CAN-01"],
            "cell_index": [0],
            "x_um": [1.0],
            "y_um": [2.0],
        }
    )
    _validate_interpretation_identifier_minimization(permitted)

    for prohibited in (
        "original_cell_identifier",
        "source_slide",
        "fov",
        "cell_ID",
        "patient_id",
    ):
        exposed = permitted.assign(**{prohibited: ["restricted"]})
        with pytest.raises(
            pipeline.AttentionNichePipelineError,
            match="prohibited source identifiers",
        ):
            _validate_interpretation_identifier_minimization(exposed)


def test_incident_medians_and_partition_codes_are_deterministic() -> None:
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    values = np.asarray([1.0, 5.0, 3.0], dtype=np.float64)

    np.testing.assert_allclose(
        _incident_median(4, pairs, values, default=0.0),
        [3.0, 2.0, 4.0, 0.0],
    )
    np.testing.assert_array_equal(
        _integer_partition(["C01-N002", "C01-N001", "C01-N002"]),
        [1, 0, 1],
    )


def test_mutual_parquet_retains_both_directions_and_locked_statistics(
    tmp_path: Path,
) -> None:
    # Directed convention is source -> receiver.  Pair (0, 1) therefore uses
    # edge positions 0 (0->1) and 1 (1->0), never a fabricated reverse edge.
    edge_index = np.asarray(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64
    )
    routing = np.asarray(
        [
            [[2.0, 3.0, 4.0, 5.0], [4.0, 5.0, 6.0, 7.0]],
            [[6.0, 7.0, 8.0, 9.0], [8.0, 9.0, 10.0, 11.0]],
        ],
        dtype=np.float32,
    )
    consensus = summarize_mutual_routing_consensus(
        edge_index,
        routing,
        n_nodes=3,
        seed_ids=(3, 8),
        mask_view_ids=(0, 1),
        require_all_edges_reciprocal=True,
    )
    retained = retain_top_k_mutual_edges(
        consensus.pairs.pair_cells,
        consensus.Mij,
        consensus.Pij,
        n_nodes=3,
        top_k=1,
    )
    path = tmp_path / "mutual.parquet"

    receipt = _write_mutual_core_table(
        path,
        alias="CAN-01",
        core_number=1,
        coordinates=np.asarray([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]),
        consensus=consensus,
        retained_graph=retained,
        row_group_size=1,
    )
    frame = pd.read_parquet(path)

    assert receipt["row_count"] == 2
    assert frame["core_number"].tolist() == [1, 1]
    assert frame[["cell_i_index", "cell_j_index"]].values.tolist() == [
        [0, 1],
        [1, 2],
    ]
    assert frame["i_to_j_edge_index_position"].tolist() == [0, 2]
    assert frame["j_to_i_edge_index_position"].tolist() == [1, 3]
    np.testing.assert_allclose(frame["distance_um"], [5.0, 5.0])
    np.testing.assert_allclose(frame["M_ij"], consensus.Mij)
    np.testing.assert_allclose(frame["support_P_ij"], consensus.Pij)
    assert frame["retained_primary"].sum() == retained.edge_count


def test_archive_receipts_drop_scratch_paths_and_index_inputs_portably() -> None:
    absolute = "/workspace/BAGM/scratch/active_runs/r_x/core_01/table.parquet"
    receipts = _portable_core_receipts(
        [
            {
                "core_number": 1,
                "outputs": {"directed_edges": absolute},
                "receipt_path": "/workspace/BAGM/scratch/active_runs/r_x/receipt.json",
                "directed_table": {
                    "path": absolute,
                    "row_count": 4,
                    "sha256": "a" * 64,
                },
                "mutual_table": {
                    "path": absolute,
                    "row_count": 2,
                    "sha256": "b" * 64,
                },
            }
        ]
    )

    assert absolute not in str(receipts)
    assert receipts[0]["canonical_outputs"]["directed_edges"] == {
        "path": "directed_attention_edges.parquet",
        "filter": "core_number == 1",
    }
    assert receipts[0]["directed_table"]["temporary_core_shard_retained"] is False
    assert _portable_file_receipts(
        {absolute: {"path": "data/raw/input.csv", "sha256": "c" * 64}}
    ) == {"data/raw/input.csv": {"path": "data/raw/input.csv", "sha256": "c" * 64}}


def test_verified_staging_cleanup_is_narrow_and_reports_removed_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "scratch" / "active_runs" / "r_x"
    work = output / "diagnostics" / "attention_niche_core_work"
    work.mkdir(parents=True)
    (work / "shard.parquet").write_bytes(b"12345")

    receipt = _remove_verified_core_work_tree(work, output_root=output)

    assert receipt == {
        "removed_after_verified_consolidation": True,
        "file_count": 1,
        "size_bytes": 5,
    }
    assert not work.exists()
    assert not (output / "diagnostics").exists()


def test_directed_export_retains_exact_seed_view_routing_without_fake_seed_sd(
    tmp_path: Path,
) -> None:
    path = tmp_path / "directed.parquet"
    routing = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    visible = np.asarray([[5.0, 6.0]], dtype=np.float32)
    channels = {
        name: np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
        for name in ("attention", "content_qk", "positional_bias", "combined_logit")
    }

    receipt = _write_directed_core_table(
        path,
        alias="CAN-01",
        core_number=1,
        seeds=(7,),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        coordinates=np.asarray([[0.0, 0.0], [3.0, 4.0]]),
        indegree=np.asarray([1, 1], dtype=np.int64),
        routing_samples=routing,
        all_visible_routing=visible,
        channel_mask_means=channels,
        row_group_size=1,
    )
    frame = pd.read_parquet(path)

    assert receipt["row_count"] == 2
    assert frame["seed_aggregated_sd"].isna().all()
    np.testing.assert_allclose(
        np.stack(frame["degree_adjusted_routing_per_seed_mask_view"]),
        [[1.0, 3.0], [2.0, 4.0]],
    )
    np.testing.assert_allclose(
        np.stack(frame["all_visible_degree_adjusted_routing_per_seed"]),
        [[5.0], [6.0]],
    )


def test_core_parquet_combination_verifies_contract_and_removes_shards(
    tmp_path: Path,
) -> None:
    work = tmp_path / "diagnostics" / "attention_niche_core_work"
    sources: list[Path] = []
    contracts: list[dict[str, object]] = []
    for core_number, alias in ((1, "CAN-01"), (9, "CAN-09")):
        core_dir = work / f"core_{core_number:02d}"
        core_dir.mkdir(parents=True)
        source = core_dir / "assignments.parquet"
        pd.DataFrame(
            {
                "core_number": [core_number, core_number],
                "core_alias": [alias, alias],
                "cell_index": [0, 1],
                "value": [1.0, 2.0],
            }
        ).to_parquet(source, index=False)
        sources.append(source)
        contracts.append(
            {
                "core_number": core_number,
                "core_alias": alias,
                "row_count": 2,
                "position_column": "cell_index",
            }
        )

    destination = tmp_path / "combined.parquet"
    receipt = _combine_parquet_files(
        sources,
        destination,
        source_contracts=contracts,
        remove_sources_after_write=True,
        staging_root=work,
    )

    assert receipt["row_count"] == 4
    assert receipt["source_contracts_verified"] is True
    assert receipt["source_shards_removed_after_streaming"] == 2
    assert all(not source.exists() for source in sources)
    assert pd.read_parquet(destination)["core_number"].tolist() == [1, 1, 9, 9]


def test_seed_aware_disk_estimate_enforces_configured_floor() -> None:
    four = _required_analysis_free_disk_gib(
        model_seed_count=4,
        minimum_free_disk_gib=40.0,
    )
    five = _required_analysis_free_disk_gib(
        model_seed_count=5,
        minimum_free_disk_gib=40.0,
    )

    assert four["required_free_gib"] >= 40.0
    assert five["required_free_gib"] > four["required_free_gib"]


def test_streamed_production_attention_reconstructs_softmax_and_edge_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge_index = np.asarray([[0, 2, 1], [1, 1, 2]], dtype=np.int64)
    indegree = np.asarray([0, 2, 1], dtype=np.int64)
    combined = np.asarray([[0.0], [np.log(3.0)], [2.0]], dtype=np.float32)
    content = combined.copy()
    bias = np.zeros_like(combined)
    attention = np.asarray([[0.25], [0.75], [1.0]], dtype=np.float32)

    def fake_stream(
        model: object,
        batch: object,
        mask: np.ndarray,
        *,
        layer: int,
        amp: bool,
        consumer: object,
    ) -> int:
        assert layer == -1
        consumer(
            1,
            3,
            np.arange(3, dtype=np.int64),
            attention,
            content,
            bias,
            combined,
        )
        return 0

    monkeypatch.setattr(pipeline, "stream_receiver_attention", fake_stream)
    routing = np.empty(3, dtype=np.float32)
    audit = _stream_one_view(
        model=SimpleNamespace(graph_layers=1),
        batch=object(),
        mask=np.zeros((3, 2), dtype=bool),
        edge_index=edge_index,
        indegree=indegree,
        routing_target=routing,
        channel_targets=None,
        amp=False,
    )

    np.testing.assert_allclose(routing, [0.5, 1.5, 1.0])
    assert audit["strict_attention_shard_count"] == 1
    assert audit["max_softmax_reconstruction_error"] < 1e-6

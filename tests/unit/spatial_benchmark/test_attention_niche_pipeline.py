from __future__ import annotations

import errno
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import spatial_benchmark.attention_niche_pipeline as pipeline
from spatial_benchmark.attention_niche_pipeline import (
    CoreJobSpec,
    _bundle_retention_tombstones,
    _combine_parquet_files,
    _incident_median,
    _integer_partition,
    _materialize_ficlone_reflink,
    _portable_core_receipts,
    _portable_file_receipts,
    _required_analysis_free_disk_gib,
    _scientific_config_without_recovery_runtime,
    _remove_verified_core_work_tree,
    _stream_one_view,
    _validate_interpretation_identifier_minimization,
    _stream_validate_recovery_edge_alignment,
    _validated_render_recovery_request,
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


def test_render_recovery_requires_exact_launcher_identity() -> None:
    assert _validated_render_recovery_request({}) is None
    recovery = {
        "mode": pipeline.RENDER_RECOVERY_MODE,
        "source_run_id": pipeline.RENDER_RECOVERY_SOURCE_RUN_ID,
        "source_queue_job_id": pipeline.RENDER_RECOVERY_SOURCE_QUEUE_JOB_ID,
        "source_failed_marker_content_sha256": (
            pipeline.RENDER_RECOVERY_FAILED_CONTENT_SHA256
        ),
        "expected_renderer_failure_signature": (
            pipeline.RENDER_RECOVERY_FAILURE_SIGNATURE
        ),
        "minimum_figure_headroom_gib": 2.0,
    }
    request = _validated_render_recovery_request({"recovery": recovery})

    assert request is not None
    assert request.source_run_id == pipeline.RENDER_RECOVERY_SOURCE_RUN_ID
    drifted = {**recovery, "source_queue_job_id": "q_wrong"}
    with pytest.raises(pipeline.AttentionNichePipelineError, match="drifted"):
        _validated_render_recovery_request({"recovery": drifted})
    with pytest.raises(pipeline.AttentionNichePipelineError, match="unrecognized"):
        _validated_render_recovery_request(
            {"recovery": {**recovery, "copy_fallback": True}}
        )


def test_recovery_launcher_is_excluded_but_metadata_is_scientific() -> None:
    source = {
        "metadata": {"analysis_mask_views": 10},
        "launcher": {"requested_gpu": "0,2,3"},
        "seed": 0,
    }
    recovery = {
        **source,
        "launcher": {
            "requested_gpu": "0,2,3",
            "recovery": {"mode": pipeline.RENDER_RECOVERY_MODE},
        },
    }

    assert _scientific_config_without_recovery_runtime(
        source
    ) == _scientific_config_without_recovery_runtime(recovery)
    metadata_drift = {
        **recovery,
        "metadata": {"analysis_mask_views": 9},
    }
    assert _scientific_config_without_recovery_runtime(
        source
    ) != _scientific_config_without_recovery_runtime(metadata_drift)


def test_recovery_materialization_uses_distinct_inode_ficlone(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"locked recovery payload" * 1024)

    receipt = _materialize_ficlone_reflink(
        source,
        destination,
        expected_sha256=pipeline.sha256_file(source),
    )

    assert receipt["method"] == "linux_ficlone_reflink"
    assert receipt["distinct_inode"] is True
    assert receipt["copy_fallback_permitted"] is False
    assert source.stat().st_ino != destination.stat().st_ino
    assert source.read_bytes() == destination.read_bytes()


def test_recovery_ficlone_failure_has_no_copy_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"locked recovery payload")

    def fail_ioctl(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "not supported")

    monkeypatch.setattr(pipeline.fcntl, "ioctl", fail_ioctl)
    with pytest.raises(pipeline.AttentionNichePipelineError, match="no copy fallback"):
        _materialize_ficlone_reflink(
            source,
            destination,
            expected_sha256=pipeline.sha256_file(source),
        )
    assert not destination.exists()


def test_recovery_streams_exact_directed_and_reciprocal_edge_alignment(
    tmp_path: Path,
) -> None:
    graph_dir = tmp_path / "graph"
    core_dir = graph_dir / "cores" / "CAN-01"
    core_dir.mkdir(parents=True)
    np.save(
        core_dir / "edge_index.npy",
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        allow_pickle=False,
    )
    pd.DataFrame(
        {
            "core_number": [1, 1],
            "core_alias": ["CAN-01", "CAN-01"],
            "edge_index_position": [0, 1],
            "source_cell_index": [0, 1],
            "receiver_cell_index": [1, 0],
            "receiver_in_degree": [1, 1],
            "head_mean_attention_mean": [0.75, 0.50],
            "degree_adjusted_routing_mean": [0.75, 0.50],
        }
    ).to_parquet(tmp_path / "directed_attention_edges.parquet", index=False)
    mutual = pd.DataFrame(
        {
            "core_number": [1],
            "core_alias": ["CAN-01"],
            "mutual_pair_position": [0],
            "cell_i_index": [0],
            "cell_j_index": [1],
            "i_to_j_edge_index_position": [0],
            "j_to_i_edge_index_position": [1],
        }
    )
    mutual.to_parquet(tmp_path / "mutual_attention_edges.parquet", index=False)
    contract = pipeline.PreparedInputContract(
        cohort_dir=tmp_path / "cohort",
        graph_dir=graph_dir,
        raw_dir=tmp_path / "raw",
        core_map_path=tmp_path / "map.csv",
        reconciliation_path=tmp_path / "policy.yaml",
        cohort_manifest={},
        graph_manifest={},
        core_records={},
        gene_names=(),
        metadata_names=(),
        pre_analysis_file_receipts={},
    )
    receipt = {
        "core_number": 1,
        "core_alias": "CAN-01",
        "cell_count": 2,
        "directed_edge_count": 2,
        "reciprocal_pair_count": 1,
    }

    audit = _stream_validate_recovery_edge_alignment(
        tmp_path,
        input_contract=contract,
        core_receipts=[receipt],
    )

    assert audit["edge_index_source_receiver_alignment_exact"] is True
    assert audit["receiver_indegree_exact"] is True
    assert audit["degree_adjustment_uses_receiver_indegree"] is True
    assert audit["every_mutual_pair_has_two_reversed_exported_edges"] is True
    assert audit["every_directed_edge_covered_by_exactly_one_mutual_pair"] is True

    mutual["j_to_i_edge_index_position"] = 0
    mutual.to_parquet(tmp_path / "mutual_attention_edges.parquet", index=False)
    with pytest.raises(
        pipeline.AttentionNichePipelineError,
        match="reciprocal-edge alignment",
    ):
        _stream_validate_recovery_edge_alignment(
            tmp_path,
            input_contract=contract,
            core_receipts=[receipt],
        )


def test_checkpoint_bundle_tombstones_require_audited_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "artifacts" / "runs" / "2026" / "08" / "r_source"
    bundle.mkdir(parents=True)
    deleted = bundle / "checkpoints" / "epoch_0025.ckpt"
    records = [
        {
            "path": str(deleted),
            "sha256": "a" * 64,
            "size_bytes": 123,
            "status": "deleted_by_retention",
        }
    ]

    assert _bundle_retention_tombstones(
        bundle, records, project_root=tmp_path
    ) == {
        "checkpoints/epoch_0025.ckpt": {
            "type": "file",
            "size": 123,
            "sha256": "a" * 64,
        }
    }
    with pytest.raises(pipeline.AttentionNichePipelineError, match="pending"):
        _bundle_retention_tombstones(
            bundle,
            [{**records[0], "status": "retention_pending"}],
            project_root=tmp_path,
        )


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

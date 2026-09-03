from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import spatial_benchmark.so1_raw_expression_clustering as analysis


SO1_CORE_NUMBERS = analysis.SO1_CORE_NUMBERS


def _floor_fixture() -> np.ndarray:
    # Library sizes 20, 2, and 60 make the floor behavior directly auditable.
    return np.asarray(
        [
            [0, 10, 10],
            [1, 1, 0],
            [10, 20, 30],
        ],
        dtype=np.int32,
    )


def _pca_fixture() -> sparse.csr_matrix:
    rng = np.random.default_rng(20260825)
    counts = rng.poisson(lam=4.0, size=(36, 7)).astype(np.int32)
    counts[:, -1] = 0  # One constant gene must be excluded, not destabilize PCA.
    counts[:, 0] += 1  # Every cell retains a positive expression total.
    transformed, _, _ = analysis.log_normalize_counts(
        counts,
        normalization_target=40.0,
        library_size_floor=20.0,
    )
    return transformed


def _normalized_score_fixture() -> np.ndarray:
    rng = np.random.default_rng(17)
    centers = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    values = np.vstack(
        [center + rng.normal(0.0, 0.035, size=(28, 4)) for center in centers]
    ).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values)


def _cell_frame(*, cells_per_core: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    global_index = 0
    for offset, core_number in enumerate(SO1_CORE_NUMBERS):
        for cell_index in range(cells_per_core):
            rows.append(
                {
                    "global_cell_index": global_index,
                    "cell_index": cell_index,
                    "cell_key": f"SO1-C{core_number:02d}:{cell_index:08d}",
                    "core_alias": f"SO1-C{core_number:02d}",
                    "core_number": core_number,
                    "x_um": float(10 * offset + cell_index),
                    "y_um": float(4 * offset + cell_index),
                }
            )
            global_index += 1
    return pd.DataFrame(rows)


def test_cpu_device_validation_fails_closed_on_cuda() -> None:
    assert analysis.validate_cpu_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="CPU|cpu"):
        analysis.validate_cpu_device("cuda")


def test_so1_locked_source_and_method_constants_are_slide_specific() -> None:
    assert analysis.SO1_CORE_NUMBERS == tuple(range(1, 15))
    assert analysis.SO1_ALIASES == tuple(
        f"SO1-C{number:02d}" for number in range(1, 15)
    )
    assert sum(analysis.EXPECTED_CELL_COUNTS_BY_CORE.values()) == 161_596
    assert analysis.EXPECTED_TOTAL_CELLS == 161_596
    assert analysis.DEFAULT_NORMALIZATION_TARGET == 162.0
    assert analysis.LABEL_PREFIX == "S1E"


def test_log_normalization_uses_library_floor_and_keeps_every_cell() -> None:
    counts = _floor_fixture()
    original = counts.copy()

    transformed, qc, receipt = analysis.log_normalize_counts(
        counts,
        normalization_target=40.0,
        library_size_floor=20.0,
    )

    assert sparse.isspmatrix_csr(transformed)
    assert transformed.shape == counts.shape
    assert transformed.nnz == np.count_nonzero(counts)
    assert np.array_equal(counts, original)
    assert np.isfinite(transformed.data).all()
    np.testing.assert_allclose(
        np.expm1(transformed.toarray()).sum(axis=1),
        np.asarray([40.0, 4.0, 40.0]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        qc["raw_library_size"], np.asarray([20.0, 2.0, 60.0])
    )
    np.testing.assert_array_equal(
        qc["normalization_denominator"], np.asarray([20.0, 20.0, 60.0])
    )
    np.testing.assert_array_equal(
        qc["below_library_size_floor"], np.asarray([False, True, False])
    )
    assert receipt["normalization_target"] == 40.0
    assert receipt["library_size_floor"] == 20.0
    assert receipt["cells_retained"] == 3
    assert receipt["cells_dropped"] == 0
    assert receipt["sparse_output"] is True


@pytest.mark.parametrize(
    "counts",
    (
        np.asarray([[1, -1], [2, 3]], dtype=np.int32),
        np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64),
        np.asarray([[1.5, 2.0], [2.0, 3.0]], dtype=np.float64),
        np.asarray([[0, 0], [2, 3]], dtype=np.int32),
        np.asarray([1, 2, 3], dtype=np.int32),
    ),
)
def test_log_normalization_rejects_invalid_raw_counts(counts: np.ndarray) -> None:
    with pytest.raises(ValueError, match="count|finite|integer|library|shape|2"):
        analysis.log_normalize_counts(
            counts,
            normalization_target=40.0,
            library_size_floor=20.0,
        )


def test_exact_scaled_sparse_pca_is_deterministic_signed_and_never_dense_nxn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _pca_fixture()
    n_cells = matrix.shape[0]
    for allocator_name in ("empty", "zeros", "ones", "full"):
        original_allocator = getattr(np, allocator_name)

        def guarded_allocator(
            shape: object,
            *args: object,
            _original: object = original_allocator,
            _name: str = allocator_name,
            **kwargs: object,
        ) -> np.ndarray:
            if isinstance(shape, tuple) and shape == (n_cells, n_cells):
                pytest.fail(f"dense N x N allocation through numpy.{_name}")
            return _original(shape, *args, **kwargs)

        monkeypatch.setattr(np, allocator_name, guarded_allocator)

    first = analysis.exact_scaled_sparse_pca(
        matrix,
        n_components=4,
        clip_value=10.0,
    )
    second = analysis.exact_scaled_sparse_pca(
        matrix,
        n_components=4,
        clip_value=10.0,
    )

    assert first.normalized_scores.shape == (n_cells, 4)
    assert first.components.shape == (matrix.shape[1], 4)
    assert first.gene_mean.shape == (matrix.shape[1],)
    assert first.gene_scale.shape == (matrix.shape[1],)
    assert first.retained_mask.dtype == np.bool_
    assert not bool(first.retained_mask[-1])
    assert np.count_nonzero(first.components[~first.retained_mask]) == 0
    assert np.isfinite(first.normalized_scores).all()
    np.testing.assert_allclose(
        np.linalg.norm(first.normalized_scores, axis=1),
        np.ones(n_cells),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        first.components.T @ first.components,
        np.eye(4),
        rtol=1e-5,
        atol=1e-6,
    )
    for component in first.components.T:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] >= 0.0
    assert np.array_equal(first.normalized_scores, second.normalized_scores)
    assert np.array_equal(first.components, second.components)
    assert first.receipt == second.receipt
    assert first.receipt["cell_by_cell_matrix_constructed"] is False
    assert first.receipt["feature_covariance_shape"] == [6, 6]
    assert first.receipt["gene_centering"] is True
    assert first.receipt["gene_scaling"] == "sample_standard_deviation_ddof_1"
    assert first.receipt["scale_clip_absolute_value"] == 10.0
    assert first.receipt["l2_normalized_after_pca"] is True


def test_sparse_pca_matches_independent_dense_scaled_and_clipped_reference() -> None:
    matrix = _pca_fixture()
    observed = analysis.exact_scaled_sparse_pca(
        matrix,
        n_components=4,
        clip_value=2.0,
    )
    dense = matrix.toarray().astype(np.float64)
    gene_mean = dense.mean(axis=0)
    gene_scale = dense.std(axis=0, ddof=1)
    retained = gene_scale > 0.0
    unbounded = (dense[:, retained] - gene_mean[retained]) / gene_scale[retained]
    assert np.any(np.abs(unbounded) > 2.0)
    scaled = np.clip(unbounded, -2.0, 2.0)
    centered = scaled - scaled.mean(axis=0)
    covariance = centered.T @ centered / float(len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    expected_components = eigenvectors[:, order[:4]]
    for column in range(expected_components.shape[1]):
        pivot = int(np.argmax(np.abs(expected_components[:, column])))
        if expected_components[pivot, column] < 0.0:
            expected_components[:, column] *= -1.0
    expected_scores = centered @ expected_components
    expected_scores /= np.linalg.norm(expected_scores, axis=1, keepdims=True)

    np.testing.assert_allclose(observed.gene_mean, gene_mean, atol=1e-12)
    np.testing.assert_allclose(observed.gene_scale, gene_scale, atol=1e-12)
    np.testing.assert_array_equal(observed.retained_mask, retained)
    np.testing.assert_allclose(
        observed.components[retained], expected_components, atol=1e-10
    )
    np.testing.assert_allclose(
        observed.normalized_scores,
        expected_scores.astype(np.float32),
        rtol=1e-5,
        atol=1e-6,
    )


def test_expression_cluster_scorer_accepts_only_scores_and_is_deterministic() -> None:
    signature = inspect.signature(analysis.cluster_expression_scores)
    assert tuple(signature.parameters) == (
        "scores",
        "n_neighbors",
        "leiden_resolution",
        "random_seed",
    )
    forbidden = {"coordinates", "metadata", "graph", "model", "core_number"}
    assert forbidden.isdisjoint(signature.parameters)
    scores = _normalized_score_fixture()

    labels_a, receipt_a, graph_a = analysis.cluster_expression_scores(
        scores,
        n_neighbors=5,
        leiden_resolution=1.0,
        random_seed=20260825,
    )
    labels_b, receipt_b, graph_b = analysis.cluster_expression_scores(
        scores,
        n_neighbors=5,
        leiden_resolution=1.0,
        random_seed=20260825,
    )

    assert labels_a.shape == (len(scores),)
    assert np.array_equal(labels_a, labels_b)
    assert np.array_equal(np.unique(labels_a), np.arange(labels_a.max() + 1))
    assert receipt_a == receipt_b
    assert np.array_equal(graph_a.edge_pairs, graph_b.edge_pairs)
    assert graph_a.edge_pairs.ndim == 2 and graph_a.edge_pairs.shape[1] == 2
    assert not np.any(graph_a.edge_pairs[:, 0] == graph_a.edge_pairs[:, 1])
    assert receipt_a["knn"] == graph_a.receipt
    assert receipt_a["knn"]["cell_by_cell_matrix_constructed"] is False
    assert receipt_a["knn"]["input_order"] == (
        "deterministic_seeded_permutation_then_mapped_to_locked_global_rows"
    )
    assert receipt_a["knn"]["core_labels_consulted_for_insertion_order"] is False
    assert receipt_a["knn"]["insertion_permutation_seed"] == (
        20260825 ^ 0x534F31
    )
    assert len(receipt_a["knn"]["insertion_permutation_sha256"]) == 64
    assert receipt_a["knn"]["maximum_explicit_pairwise_array_shape"][1] < len(
        scores
    )
    assert receipt_a["leiden"]["resolution"] == 1.0
    serialized = json.dumps(receipt_a, sort_keys=True).lower()
    for prohibited in ("coordinate", "metadata", "checkpoint", "spatial graph"):
        assert prohibited not in serialized


def test_expression_cluster_summaries_use_distinct_so1_prefix() -> None:
    frame = _cell_frame()
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), len(SO1_CORE_NUMBERS))

    summary, composition, dominated = analysis.cluster_summary_tables(
        labels,
        frame,
        prefix="S1E",
    )

    assert summary["cluster"].tolist() == ["S1E0", "S1E1"]
    assert summary["size"].tolist() == [14, 14]
    assert set(composition["core_number"]) == set(SO1_CORE_NUMBERS)
    assert len(composition) == 2 * len(SO1_CORE_NUMBERS)
    assert dominated == []


def test_low_count_warning_is_structured_and_derived_from_qc_summary() -> None:
    qc = pd.DataFrame(
        {
            "cluster": ["S1E0", "S1E1"],
            "cluster_number": [0, 1],
            "size": [80, 20],
            "below_library_size_floor_count": [4, 6],
            "below_library_size_floor_proportion": [0.05, 0.30],
        }
    )

    warning = analysis._low_count_depth_warning(
        qc,
        below_threshold_cells=10,
        threshold_transcripts=20.0,
    )

    assert warning["threshold_transcripts"] == 20.0
    assert warning["cohort_below_threshold_cells"] == 10
    assert warning["flagged_clusters"] == [
        {
            "cluster": "S1E1",
            "below_threshold_cells": 6,
            "cluster_cells": 20,
            "fraction_below_threshold": 0.30,
        }
    ]


def test_panel_order_and_shared_expression_palette_are_locked() -> None:
    palette = {
        "S1E0": "#0072B2",
        "S1E1": "#D55E00",
        "S1E2": "#009E73",
    }
    specification = analysis.spatial_plot_spec(palette)

    assert analysis.requested_panel_order() == tuple(range(1, 15))
    assert tuple(specification["panel_order"]) == tuple(range(1, 15))
    assert specification["grid_shape"] == [3, 5]
    assert specification["equal_aspect"] is True
    assert specification["invert_y_axis"] is True
    assert specification["core_numbers_identifiable"] is True
    assert specification["one_shared_joint_cluster_palette"] is True
    assert specification["palette"] == palette
    assert specification["panel_title_template"].format(core_number=11) == (
        "SO1 Core 11"
    )
    serialized = json.dumps(specification, sort_keys=True).lower()
    assert "contextual" not in serialized
    assert "intrinsic" not in serialized

    first = analysis.deterministic_expression_palette(24)
    second = analysis.deterministic_expression_palette(24)
    assert first == second
    assert list(first) == [f"S1E{index}" for index in range(24)]
    assert len(set(first.values())) == 24


def test_raw_loader_never_accesses_nonexpression_npz_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = ("TEST-C01", "TEST-C02")
    core_numbers = (1, 2)
    cells_by_core = {1: 2, 2: 2}
    counts = np.asarray([[10, 10, 0], [10, 10, 20]], dtype=np.int32)
    source_records: list[dict[str, object]] = []
    core_dir = tmp_path / "cores"
    core_dir.mkdir()
    for alias, core_number in zip(aliases, core_numbers, strict=True):
        # Object arrays fail under allow_pickle=False if code touches them.
        poison = np.asarray([{"must_not_load": True}], dtype=object)
        np.savez(
            core_dir / f"{alias}.npz",
            expression_counts=counts,
            target_expression=poison,
            node_covariates=poison,
            coordinates_um=poison,
        )
        source_records.append(
            {
                "component_checksums": {
                    "expression_counts": analysis._source_array_sha256(
                        "expression_counts", counts
                    )
                }
            }
        )
    monkeypatch.setattr(analysis, "SO1_ALIASES", aliases)
    monkeypatch.setattr(analysis, "SO1_CORE_NUMBERS", core_numbers)
    monkeypatch.setattr(analysis, "EXPECTED_CELL_COUNTS_BY_CORE", cells_by_core)
    monkeypatch.setattr(analysis, "EXPECTED_TOTAL_CELLS", 4)
    monkeypatch.setattr(analysis, "EXPECTED_N_GENES", 3)
    inputs = analysis.RawExpressionInputs(
        cohort_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        manifest={},
        gene_names=("G0", "G1", "G2"),
        source_records=tuple(source_records),
    )

    matrix, qc, receipt = analysis._load_and_normalize_raw_counts(
        inputs=inputs,
        normalization_target=30.0,
        library_size_floor=20.0,
    )

    assert matrix.shape == (4, 3)
    assert len(qc) == 4
    assert receipt["source_array_access"] == ["expression_counts"]
    assert receipt["prohibited_arrays_accessed"] is False
    assert receipt["joint_core_order"] == [1, 2]


def test_completed_preprocessing_stage_is_verified_and_reused_without_reloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "preprocessing" / "preprocessing_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    expected = {"schema": analysis.PREPROCESSING_SCHEMA, "status": "complete"}
    inputs = analysis.RawExpressionInputs(
        cohort_dir=tmp_path / "cohort",
        manifest_path=tmp_path / "cohort" / "manifest.json",
        manifest={},
        gene_names=(),
        source_records=(),
    )
    verified: list[dict[str, object]] = []
    monkeypatch.setattr(analysis, "_read_json", lambda *_args, **_kwargs: expected)

    def record_verification(**kwargs: object) -> None:
        verified.append(kwargs)

    monkeypatch.setattr(analysis, "_verify_preprocessing_receipt", record_verification)
    monkeypatch.setattr(
        analysis,
        "_load_and_normalize_raw_counts",
        lambda **_kwargs: pytest.fail("completed preprocessing was recomputed"),
    )

    observed = analysis.prepare_expression_representation(
        inputs=inputs,
        output_root=tmp_path,
        configuration={"locked": True},
    )

    assert observed is expected
    assert len(verified) == 1
    assert verified[0]["receipt"] is expected
    assert verified[0]["configuration"] == {"locked": True}


def _required_final_files() -> set[str]:
    required = {
        "README.md",
        "preprocessing/preprocessing_manifest.json",
        "preprocessing/expression_pca_l2_normalized.npy",
        "preprocessing/expression_gene_statistics.npz",
        "preprocessing/expression_preprocessing_parameters.json",
        "preprocessing/gene_names.json",
        "preprocessing/cell_source_qc.parquet",
        "clustering/clustering_manifest.json",
        "clustering/expression_labels.npy",
        "clustering/expression_knn_undirected_edges.npy",
        "clustering/expression_clustering_parameters.json",
        "clustering/expression_palette.json",
        "tables/cell_expression_clusters.parquet",
        "tables/expression_cluster_summary.csv",
        "tables/expression_cluster_core_composition.csv",
        "tables/expression_cluster_qc_summary.csv",
        "figures/figure_manifest.json",
        "figures/so1_raw_expression_leiden_resolution_1p0_spatial_14cores.png",
        "figures/so1_raw_expression_leiden_resolution_1p0_spatial_14cores.pdf",
        "provenance/analysis_code_provenance.json",
    }
    required.update(
        "figures/per_core/"
        f"so1_core_{core}_raw_expression_leiden_resolution_1p0.png"
        for core in SO1_CORE_NUMBERS
    )
    return required


def test_final_manifest_validates_required_checksums_and_detects_tampering(
    tmp_path: Path,
) -> None:
    for relative in sorted(_required_final_files()):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    low_count_warning = {
        "threshold_transcripts": 20.0,
        "cohort_below_threshold_cells": 0,
        "flag_rule": (
            "cluster_contains_at_least_50pct_of_all_cohort_cells_below_threshold"
        ),
        "flagged_clusters": [],
    }
    (tmp_path / "clustering" / "clustering_manifest.json").write_text(
        json.dumps({"low_count_depth_warning": low_count_warning}),
        encoding="utf-8",
    )
    configuration = {"locked": True}
    exclusion_audit = {
        "checkpoint_used": False,
        "trained_model_used": False,
        "h0_used": False,
        "hL_used": False,
        "delta_h_used": False,
        "metadata_used": False,
        "spatial_graph_used": False,
        "relative_geometry_used": False,
        "coordinates_used_before_labels_frozen": False,
        "core_identity_used_before_labels_frozen": False,
        "clinical_or_vendor_labels_used": False,
    }
    manifest = analysis._receipt_with_self_hash(
        {
            "schema": analysis.ANALYSIS_SCHEMA,
            "status": "complete",
            "analysis_id": analysis.ANALYSIS_ID,
            "configuration": configuration,
            "core_order": list(SO1_CORE_NUMBERS),
            "core_cell_counts": {
                str(core): analysis.EXPECTED_CELL_COUNTS_BY_CORE[core]
                for core in SO1_CORE_NUMBERS
            },
            "total_cells": analysis.EXPECTED_TOTAL_CELLS,
            "gene_count": analysis.EXPECTED_N_GENES,
            "source_artifacts": analysis._expected_source_file_records(),
            "stage_manifests": {
                "preprocessing": analysis._file_record(
                    tmp_path / "preprocessing" / "preprocessing_manifest.json"
                ),
                "clustering": analysis._file_record(
                    tmp_path / "clustering" / "clustering_manifest.json"
                ),
                "figures": analysis._file_record(
                    tmp_path / "figures" / "figure_manifest.json"
                ),
            },
            "input_exclusion_audit": exclusion_audit,
            "low_count_depth_warning": low_count_warning,
            "files": analysis._file_manifest(tmp_path),
        }
    )

    analysis._verify_final_manifest(
        output_root=tmp_path,
        manifest=manifest,
        configuration=configuration,
    )
    (tmp_path / "README.md").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum changed"):
        analysis._verify_final_manifest(
            output_root=tmp_path,
            manifest=manifest,
            configuration=configuration,
        )


def test_public_bundle_verifier_rejects_nonprimary_configuration(
    tmp_path: Path,
) -> None:
    configuration = {
        **analysis._validate_locked_parameters(
            normalization_target=analysis.DEFAULT_NORMALIZATION_TARGET,
            library_size_floor=analysis.DEFAULT_LIBRARY_SIZE_FLOOR,
            scale_clip=analysis.DEFAULT_SCALE_CLIP,
            pca_components=analysis.DEFAULT_PCA_COMPONENTS,
            n_neighbors=analysis.DEFAULT_N_NEIGHBORS,
            leiden_resolution=analysis.DEFAULT_LEIDEN_RESOLUTION,
            random_seed=analysis.DEFAULT_RANDOM_SEED,
        ),
        "figure_dpi": 300,
    }
    changed = {**configuration, "normalization_target": 999.0}
    with pytest.raises(ValueError, match="locked primary configuration"):
        analysis.verify_so1_raw_expression_clustering_bundle(
            output_root=tmp_path,
            manifest={"configuration": changed},
        )


def test_figure_receipt_is_bound_to_resolution_and_dpi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "_verify_stage_files", lambda *_args, **_kwargs: None)
    receipt = analysis._receipt_with_self_hash(
        {
            "schema": analysis.FIGURE_SCHEMA,
            "status": "complete",
            "clustering_manifest_sha256": "a" * 64,
            "leiden_resolution": 1.0,
            "dpi": 300,
            "point_count": analysis.EXPECTED_TOTAL_CELLS,
            "plot_specification": {
                "panel_order": list(SO1_CORE_NUMBERS),
                "grid_shape": [3, 5],
            },
            "files": {f"figure_{index}.png": {} for index in range(16)},
        }
    )
    analysis._verify_figure_receipt(
        output_root=tmp_path,
        receipt=receipt,
        clustering_manifest_sha256="a" * 64,
        resolution=1.0,
        dpi=300,
    )
    with pytest.raises(ValueError, match="identity"):
        analysis._verify_figure_receipt(
            output_root=tmp_path,
            receipt=receipt,
            clustering_manifest_sha256="a" * 64,
            resolution=1.0,
            dpi=600,
        )


def test_raw_expression_cli_defaults_and_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spatial_benchmark.cli as cli
    from spatial_benchmark.paths import ProjectPaths

    parser = cli.build_parser()
    arguments = parser.parse_args(["analyze-so1-raw-expression-clusters"])
    assert arguments.normalization_target == 162.0
    assert arguments.library_size_floor == 20.0
    assert arguments.scale_clip == 10.0
    assert arguments.pca_components == 50
    assert arguments.n_neighbors == 30
    assert arguments.leiden_resolution == 1.0
    assert arguments.random_seed == 20260825
    assert arguments.device == "cpu"
    assert arguments.dpi == 300
    assert arguments.output_dir is None

    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "complete", "analysis_id": "synthetic"}

    monkeypatch.setattr(analysis, "run_so1_raw_expression_clustering", fake_run)
    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})
    result = cli._dispatch(arguments, registry=object(), paths=paths)

    assert result == {"status": "complete", "analysis_id": "synthetic"}
    assert observed == {
        "paths": paths,
        "cohort_dir": None,
        "normalization_target": 162.0,
        "library_size_floor": 20.0,
        "scale_clip": 10.0,
        "pca_components": 50,
        "n_neighbors": 30,
        "leiden_resolution": 1.0,
        "random_seed": 20260825,
        "device": "cpu",
        "dpi": 300,
        "output_dir": None,
    }

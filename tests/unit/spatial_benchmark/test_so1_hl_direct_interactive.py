from __future__ import annotations

import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

import spatial_benchmark.so1_hl_direct_interactive as interactive
import spatial_benchmark.so1_model_embedding_clustering as producer
from spatial_benchmark.paths import ProjectPaths


def _frame(cluster_count: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for core_offset, core_number in enumerate(interactive.SO1_CORE_NUMBERS):
        for cell_index in range(cluster_count):
            rows.append(
                {
                    "cell_index": cell_index,
                    "global_cell_index": len(rows),
                    "cell_key": f"private-so1-{core_number}-{cell_index}",
                    "core_alias": f"SO1-C{core_number:02d}",
                    "core_number": core_number,
                    "x_um": float(core_offset * 13 + cell_index * 2 + 0.125),
                    "y_um": float(core_offset * 7 + cell_index * 3 + 0.25),
                    "contextual_cluster": f"S1C{cell_index}",
                    "hL": [float(cell_index), float(core_number)],
                    "metadata": "private",
                }
            )
    return pd.DataFrame(rows)


def _palette(cluster_count: int = 4) -> dict[str, str]:
    return interactive.deterministic_contextual_palette(cluster_count)


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        project_root=root,
        config_root=root / "configs",
        data_root=root / "data",
        artifact_root=root / "artifacts",
        state_root=root / "state",
        scratch_root=root / "scratch",
        cache_root=root / "cache",
        export_root=root / "exports",
        report_root=root / "reports",
        result_root=root / "results",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_synthetic_counts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cells_per_core: int,
) -> None:
    counts = {
        core_number: cells_per_core
        for core_number in interactive.SO1_CORE_NUMBERS
    }
    monkeypatch.setattr(interactive, "EXPECTED_CELL_COUNTS_BY_CORE", counts)
    monkeypatch.setattr(
        interactive,
        "EXPECTED_TOTAL_CELLS",
        cells_per_core * len(interactive.SO1_CORE_NUMBERS),
    )


def _write_source(
    paths: ProjectPaths,
    *,
    run_id: str,
    cluster_count: int,
) -> tuple[Path, pd.DataFrame, dict[str, str]]:
    root = (
        paths.report_root
        / "analyses"
        / interactive.DEFAULT_SOURCE_REPORT
        / run_id
    )
    table_path = root / interactive.SOURCE_TABLE_RELATIVE_PATH
    palette_path = root / interactive.SOURCE_PALETTE_RELATIVE_PATH
    figure_path = root / interactive.SOURCE_FIGURE_RELATIVE_PATH
    table_path.parent.mkdir(parents=True, exist_ok=True)
    palette_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    frame = _frame(cluster_count=cluster_count)
    palette = _palette(cluster_count=cluster_count)
    frame.to_parquet(table_path, index=False)
    palette_path.write_text(
        json.dumps(
            {
                "schema": "so1_contextual_palette_v1",
                "deterministic": True,
                "colors": palette,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    figure_path.write_bytes(b"synthetic static png")
    cluster_sizes = frame["contextual_cluster"].value_counts()

    extraction_manifest_path = root / "embeddings" / "extraction_manifest.json"
    clustering_manifest_path = root / "clustering" / "clustering_manifest.json"
    figure_manifest_path = root / "figures" / "figure_manifest.json"
    extraction_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    extraction_manifest_path.write_text(
        json.dumps(
            interactive._receipt_with_self_hash(
                {
                    "schema": producer.EXTRACTION_SCHEMA,
                    "status": "complete",
                    "run_id": run_id,
                    "core_order": list(interactive.SO1_CORE_NUMBERS),
                    "total_cells": len(frame),
                    "embedding_shapes": {
                        "h0": [
                            len(frame),
                            interactive.EXPECTED_EMBEDDING_DIMENSION,
                        ],
                        "hL": [
                            len(frame),
                            interactive.EXPECTED_EMBEDDING_DIMENSION,
                        ],
                    },
                }
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    clustering_manifest_path.write_text(
        json.dumps(
            interactive._receipt_with_self_hash(
                {
                    "schema": producer.CLUSTERING_SCHEMA,
                    "status": "complete",
                    "extraction_manifest_sha256": _sha256(
                        extraction_manifest_path
                    ),
                    "cluster_counts": {
                        "intrinsic": cluster_count,
                        "contextual": cluster_count,
                    },
                    "cluster_size_ranges": {
                        "intrinsic": [
                            int(cluster_sizes.min()),
                            int(cluster_sizes.max()),
                        ],
                        "contextual": [
                            int(cluster_sizes.min()),
                            int(cluster_sizes.max()),
                        ],
                    },
                }
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    figure_manifest_path.write_text(
        json.dumps(
            interactive._receipt_with_self_hash(
                {
                    "schema": producer.FIGURE_SCHEMA,
                    "status": "complete",
                    "clustering_manifest_sha256": _sha256(
                        clustering_manifest_path
                    ),
                    "dpi": 300,
                    "point_count": len(frame),
                    "combined_figure_pairs": 3,
                    "per_core_png_count": 42,
                    "plot_specification": producer.spatial_plot_spec(),
                    "files": {
                        interactive.SOURCE_FIGURE_RELATIVE_PATH.as_posix(): (
                            interactive._file_record(figure_path)
                        )
                    },
                }
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    configuration = {
        "pipeline": interactive.SOURCE_PIPELINE_KIND,
        "representations": ["h0", "hL"],
        "pca": False,
        "mean_center": False,
        "l2_normalize_for_cosine": True,
        "n_neighbors": interactive.DEFAULT_N_NEIGHBORS,
        "distance_metric": "cosine",
        "knn_implementation": "faiss.IndexHNSWFlat",
        "knn_symmetrization": "undirected_union_unweighted",
        "seeded_insertion_permutation_independent_of_core": True,
        "minimum_mean_exact_recall_at_k": 0.90,
        "leiden_resolution": interactive.DEFAULT_LEIDEN_RESOLUTION,
        "random_seed": interactive.DEFAULT_RANDOM_SEED,
        "cluster_sort": (
            "descending_size_then_minimum_global_cell_index_then_raw_id"
        ),
        "joint_core_order": list(interactive.SO1_CORE_NUMBERS),
        "joint_cell_count": len(frame),
        "intrinsic_label_prefix": "S1I",
        "contextual_label_prefix": interactive.LABEL_PREFIX,
        "independent_representation_graphs": True,
        "cross_core_embedding_neighbors_permitted": True,
        "spatial_training_graph_reused_for_clustering": False,
        "dense_cell_by_cell_matrix_constructed": False,
        "device": "cpu",
    }
    manifest = interactive._receipt_with_self_hash(
        {
            "schema": interactive.SOURCE_SCHEMA,
            "status": "complete",
            "run_id": run_id,
            "campaign_id": "synthetic-campaign",
            "core_order": list(interactive.SO1_CORE_NUMBERS),
            "core_cell_counts": {
                str(core_number): int(
                    (frame["core_number"] == core_number).sum()
                )
                for core_number in interactive.SO1_CORE_NUMBERS
            },
            "total_cells": len(frame),
            "configuration": configuration,
            "embedding_shapes": {
                "h0": [len(frame), interactive.EXPECTED_EMBEDDING_DIMENSION],
                "hL": [len(frame), interactive.EXPECTED_EMBEDDING_DIMENSION],
            },
            "stage_manifests": {
                "extraction": interactive._file_record(
                    extraction_manifest_path
                ),
                "clustering": interactive._file_record(
                    clustering_manifest_path
                ),
                "figures": interactive._file_record(figure_manifest_path),
            },
            "cluster_counts": {
                "intrinsic": cluster_count,
                "contextual": cluster_count,
            },
            "cluster_size_ranges": {
                "intrinsic": [
                    int(cluster_sizes.min()),
                    int(cluster_sizes.max()),
                ],
                "contextual": [
                    int(cluster_sizes.min()),
                    int(cluster_sizes.max()),
                ]
            },
            "combined_figures": {
                "intrinsic_png": (
                    "figures/intrinsic_direct_h0_leiden_resolution_1p0_"
                    "spatial_14cores.png"
                ),
                "intrinsic_pdf": (
                    "figures/intrinsic_direct_h0_leiden_resolution_1p0_"
                    "spatial_14cores.pdf"
                ),
                "contextual_png": (
                    interactive.SOURCE_FIGURE_RELATIVE_PATH.as_posix()
                ),
                "contextual_pdf": (
                    "figures/contextual_direct_hl_leiden_resolution_1p0_"
                    "spatial_14cores.pdf"
                ),
                "delta_png": "figures/delta_h_l2_spatial_14cores.png",
                "delta_pdf": "figures/delta_h_l2_spatial_14cores.pdf",
            },
            "files": interactive._file_manifest(root),
        }
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return root, frame, palette


def _install_synthetic_producer_verifier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_root: Path,
    calls: list[Path] | None = None,
) -> None:
    """Stand in only for the producer's deep external/binary verification.

    The synthetic report preserves the producer's final and stage-manifest
    shapes, while the real producer verifier is tested in its own test module.
    This hook lets the consumer test prove that it delegates first and then
    validates only fields it actually consumes.
    """

    def _verify(output_root: str | Path) -> dict[str, object]:
        resolved = Path(output_root).expanduser().resolve(strict=True)
        assert resolved == expected_root.resolve(strict=True)
        if calls is not None:
            calls.append(resolved)
        manifest = json.loads(
            (resolved / "manifest.json").read_text(encoding="utf-8")
        )
        interactive._verify_self_hash(
            manifest, label="synthetic producer-shaped final manifest"
        )
        return manifest

    monkeypatch.setattr(
        producer,
        "verify_so1_model_embedding_clustering",
        _verify,
    )


class _HTMLProbe(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canvas_cores: list[str | None] = []
        self.cluster_buttons: list[str | None] = []
        self.external_assets: list[tuple[str, str]] = []
        self.csp: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = dict(attrs)
        if tag == "canvas":
            self.canvas_cores.append(values.get("data-core"))
        if tag == "button" and "cluster-chip" in str(
            values.get("class", "")
        ).split():
            self.cluster_buttons.append(values.get("data-cluster"))
        if tag == "meta" and str(values.get("http-equiv", "")).lower() == (
            "content-security-policy"
        ):
            self.csp.append(str(values.get("content", "")))
        asset = {"script": "src", "link": "href", "img": "src"}.get(tag)
        if asset and values.get(asset):
            self.external_assets.append((tag, str(values[asset])))


def test_frame_requires_exact_so1_order_cell_indices_and_s1c_labels() -> None:
    frame = _frame(cluster_count=4)
    palette = _palette(cluster_count=4)
    validated = interactive.validate_interactive_frame(frame, palette)

    assert tuple(validated["core_number"].drop_duplicates()) == (
        interactive.SO1_CORE_NUMBERS
    )
    assert sorted(set(validated["contextual_cluster"])) == [
        "S1C0",
        "S1C1",
        "S1C2",
        "S1C3",
    ]

    missing = frame.loc[frame["core_number"] != 14]
    with pytest.raises(ValueError, match="14 SO1 cores|locked order"):
        interactive.validate_interactive_frame(missing, palette)

    reordered = frame.copy()
    reordered.loc[reordered["core_number"] == 1, "cell_index"] = [1, 0, 2, 3]
    with pytest.raises(ValueError, match="ordered contiguously"):
        interactive.validate_interactive_frame(reordered, palette)

    wrong_namespace = frame.copy()
    wrong_namespace["contextual_cluster"] = wrong_namespace[
        "contextual_cluster"
    ].str.replace("S1C", "C", regex=False)
    with pytest.raises(ValueError, match="S1C<number>|S1C0"):
        interactive.validate_interactive_frame(wrong_namespace, palette)


def test_payload_round_trips_exact_source_and_excludes_private_fields() -> None:
    frame = _frame(cluster_count=5)
    palette = _palette(cluster_count=5)
    payload = interactive.build_interactive_payload(frame, palette)

    assert payload["core_order"] == list(interactive.SO1_CORE_NUMBERS)
    assert payload["clusters"] == [f"S1C{index}" for index in range(5)]
    assert payload["coordinate_orientation"] == "low_y_at_top"
    assert payload["point_count"] == len(frame)
    interactive._verify_payload_matches_source(payload, frame, palette)

    serialized = json.dumps(payload, separators=(",", ":"))
    for prohibited in (
        "cell_index",
        "global_cell_index",
        "cell_key",
        "core_alias",
        "hL",
        "metadata",
        "checkpoint",
        "source_path",
        "/workspace",
    ):
        assert prohibited not in serialized

    tampered = json.loads(json.dumps(payload))
    first = tampered["cores"][0]
    codes = bytearray(base64.b64decode(first["cluster_codes_b64"]))
    codes[0] = 1
    first["cluster_codes_b64"] = base64.b64encode(codes).decode("ascii")
    with pytest.raises(ValueError, match="differs from the source table"):
        interactive._verify_payload_matches_source(tampered, frame, palette)


def test_html_is_offline_three_by_five_interactive_and_private(
    tmp_path: Path,
) -> None:
    frame = _frame(cluster_count=4)
    palette = _palette(cluster_count=4)
    table = tmp_path / "table.parquet"
    source_manifest = tmp_path / "source.json"
    palette_path = tmp_path / "palette.json"
    static_png = tmp_path / "static.png"
    output = tmp_path / interactive.HTML_FILENAME
    frame.to_parquet(table, index=False)
    source_manifest.write_text('{"synthetic":true}\n', encoding="utf-8")
    palette_path.write_text(json.dumps({"colors": palette}), encoding="utf-8")
    static_png.write_bytes(b"png")

    receipt = interactive.render_interactive_html(
        frame=frame,
        palette=palette,
        output_path=output,
        source_table_path=table,
        source_manifest_path=source_manifest,
        source_palette_path=palette_path,
        source_figure_path=static_png,
    )
    html = output.read_text(encoding="utf-8")
    probe = _HTMLProbe()
    probe.feed(html)

    assert output.name == interactive.HTML_FILENAME
    assert probe.canvas_cores == [
        str(number) for number in interactive.SO1_CORE_NUMBERS
    ]
    assert probe.cluster_buttons == ["S1C0", "S1C1", "S1C2", "S1C3"]
    assert probe.external_assets == []
    assert probe.csp and all("connect-src 'none'" in value for value in probe.csp)
    assert "grid-template-columns: repeat(3" in html
    assert "SO1 Core 1" in html and "SO1 Core 14" in html
    assert "SO2 Core" not in html
    assert "BAGM_SO2_VIEWER" not in html
    assert "BAGM_SO1_HL_VIEWER" in html
    assert "PCA, mean-centering, and feature projection were not used" in html
    for interaction in (
        "cluster-chip",
        "selectedCluster",
        "show-all",
        "Escape",
        "wheel",
        "pointerdown",
        "dblclick",
        "export-png",
    ):
        assert interaction in html
    for prohibited in (
        frame.loc[0, "cell_key"],
        "cell_index",
        "global_cell_index",
        "checkpoint_sha256",
        ".ckpt",
        "/workspace",
        "fetch(",
        "XMLHttpRequest",
    ):
        assert prohibited not in html
    assert receipt["browser_payload_sha256"] == (
        interactive._browser_payload_sha256(
            interactive._extract_browser_payload(html)
        )
    )
    assert receipt["source_table_sha256"] == _sha256(table)
    assert receipt["source_manifest_sha256"] == _sha256(source_manifest)
    assert receipt["cell_index_included"] is False
    assert receipt["embedding_values_included"] is False
    assert receipt["checkpoint_data_included"] is False


def test_transfer_zip_is_deterministic_single_member_and_identical(
    tmp_path: Path,
) -> None:
    html = tmp_path / interactive.HTML_FILENAME
    html.write_text("<!doctype html><title>SO1 direct hL</title>\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_receipt = interactive._write_transfer_zip(
        html_path=html, zip_path=first
    )
    second_receipt = interactive._write_transfer_zip(
        html_path=html, zip_path=second
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["zip_sha256"] == second_receipt["zip_sha256"]
    assert first_receipt["member_count"] == 1
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [interactive.HTML_FILENAME]
        assert archive.infolist()[0].date_time == (1980, 1, 1, 0, 0, 0)
        assert archive.read(interactive.HTML_FILENAME) == html.read_bytes()


def test_source_loader_requires_checksums_direct_hl_flags_and_static_orientation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_counts(monkeypatch, cells_per_core=3)
    paths = _paths(tmp_path)
    run_id = "synthetic-run"
    root, frame, palette = _write_source(
        paths, run_id=run_id, cluster_count=3
    )
    producer_calls: list[Path] = []
    _install_synthetic_producer_verifier(
        monkeypatch,
        expected_root=root,
        calls=producer_calls,
    )

    source = interactive.load_verified_so1_direct_hl_source(
        root, run_id=run_id
    )
    assert producer_calls == [root.resolve(strict=True)]
    pd.testing.assert_frame_equal(
        source.frame.reset_index(drop=True),
        frame.loc[:, list(interactive._SOURCE_TABLE_COLUMNS)].reset_index(drop=True),
    )
    assert source.palette == palette

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["configuration"]["pipeline"] == (
        producer._clustering_configuration(
            n_neighbors=interactive.DEFAULT_N_NEIGHBORS,
            leiden_resolution=interactive.DEFAULT_LEIDEN_RESOLUTION,
            random_seed=interactive.DEFAULT_RANDOM_SEED,
        )["pipeline"]
    )
    assert interactive.SOURCE_PIPELINE_KIND == manifest["configuration"]["pipeline"]
    assert manifest["embedding_shapes"]["hL"] == [
        len(frame),
        interactive.EXPECTED_EMBEDDING_DIMENSION,
    ]
    assert manifest["cluster_counts"]["contextual"] == 3
    assert manifest["combined_figures"]["contextual_png"] == (
        "figures/contextual_direct_hl_leiden_resolution_1p0_spatial_14cores.png"
    )
    assert "analysis_parameters" not in manifest
    assert "hL_shape" not in manifest
    assert "contextual_cluster_count" not in manifest
    assert "plotting" not in manifest
    content = dict(manifest)
    content.pop("manifest_content_sha256")
    content["configuration"]["pca"] = True
    manifest_path.write_text(
        json.dumps(interactive._receipt_with_self_hash(content), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="direct-hL|no-PCA"):
        interactive.load_verified_so1_direct_hl_source(root, run_id=run_id)

    content["configuration"]["pca"] = False
    figure_manifest_path = root / "figures" / "figure_manifest.json"
    figure_manifest = json.loads(
        figure_manifest_path.read_text(encoding="utf-8")
    )
    figure_content = dict(figure_manifest)
    figure_content.pop("manifest_content_sha256")
    figure_content["plot_specification"]["invert_y_axis"] = False
    figure_manifest_path.write_text(
        json.dumps(
            interactive._receipt_with_self_hash(figure_content),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    content["stage_manifests"]["figures"] = interactive._file_record(
        figure_manifest_path
    )
    content["files"]["figures/figure_manifest.json"] = (
        interactive._file_record(figure_manifest_path)
    )
    manifest_path.write_text(
        json.dumps(interactive._receipt_with_self_hash(content), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="figure-stage contract|orientation"):
        interactive.load_verified_so1_direct_hl_source(root, run_id=run_id)


def test_runner_binds_exact_sources_resumes_and_detects_output_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_counts(monkeypatch, cells_per_core=3)
    paths = _paths(tmp_path)
    run_id = "synthetic-run"
    source_root, _, _ = _write_source(
        paths, run_id=run_id, cluster_count=3
    )
    producer_calls: list[Path] = []
    _install_synthetic_producer_verifier(
        monkeypatch,
        expected_root=source_root,
        calls=producer_calls,
    )

    result = interactive.run_so1_hl_direct_interactive(
        paths=paths, run_id=run_id
    )
    rerun = interactive.run_so1_hl_direct_interactive(
        paths=paths, run_id=run_id
    )

    assert result == rerun
    assert producer_calls == [
        source_root.resolve(strict=True),
        source_root.resolve(strict=True),
    ]
    assert Path(result["output_root"]) == (
        paths.report_root
        / "analyses"
        / interactive.DEFAULT_OUTPUT_REPORT
        / run_id
    )
    assert Path(result["html"]).name == interactive.HTML_FILENAME
    assert result["cluster_labels"] == ["S1C0", "S1C1", "S1C2"]
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["source_artifacts"]["analysis_manifest"] == (
        interactive._file_record(source_root / "manifest.json")
    )
    assert manifest["source_artifacts"]["cluster_table"] == (
        interactive._file_record(
            source_root / interactive.SOURCE_TABLE_RELATIVE_PATH
        )
    )
    assert manifest["execution"]["gpu_used"] is False
    assert manifest["sharing"]["cell_index_included"] is False

    Path(result["html"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|output"):
        interactive.run_so1_hl_direct_interactive(paths=paths, run_id=run_id)


def test_runner_detects_source_table_checksum_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_counts(monkeypatch, cells_per_core=2)
    paths = _paths(tmp_path)
    run_id = "source-drift"
    source_root, _, _ = _write_source(
        paths, run_id=run_id, cluster_count=2
    )
    _install_synthetic_producer_verifier(
        monkeypatch,
        expected_root=source_root,
    )
    table = source_root / interactive.SOURCE_TABLE_RELATIVE_PATH
    table.write_bytes(table.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        interactive.run_so1_hl_direct_interactive(paths=paths, run_id=run_id)

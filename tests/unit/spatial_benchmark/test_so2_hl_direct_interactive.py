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

import spatial_benchmark.so2_hl_direct_interactive as interactive
import spatial_benchmark.so2_hl_direct_clustering as direct_clustering
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.so2_pooled_full_core import SO2_CORE_NUMBERS


def _frame(cluster_count: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    global_index = 0
    for core_offset, core_number in enumerate(SO2_CORE_NUMBERS):
        for cluster_number in range(cluster_count):
            rows.append(
                {
                    "global_cell_index": global_index,
                    "cell_index": cluster_number,
                    "cell_key": f"private-{core_number}-{cluster_number}",
                    "core_alias": f"SO2-C{core_number}",
                    "core_number": core_number,
                    "x_um": float(core_offset + cluster_number * 10 + 0.125),
                    "y_um": float(core_offset * 2 + cluster_number * 5 + 0.25),
                    "contextual_direct_cluster_number": cluster_number,
                    "contextual_direct_cluster": f"D{cluster_number}",
                }
            )
            global_index += 1
    return pd.DataFrame(rows)


def _palette(cluster_count: int = 4) -> dict[str, str]:
    return {
        f"D{index}": f"#{(0x123456 + index * 0x193B71) % 0x1000000:06X}"
        for index in range(cluster_count)
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        if tag == "button" and "cluster-chip" in str(values.get("class", "")).split():
            self.cluster_buttons.append(values.get("data-cluster"))
        if tag == "meta" and str(values.get("http-equiv", "")).lower() == (
            "content-security-policy"
        ):
            self.csp.append(str(values.get("content", "")))
        asset_attribute = {"script": "src", "link": "href", "img": "src"}.get(tag)
        if asset_attribute and values.get(asset_attribute):
            self.external_assets.append((tag, str(values[asset_attribute])))


def test_dynamic_d_labels_and_complete_core_order_are_required() -> None:
    frame = _frame(cluster_count=4)
    palette = _palette(cluster_count=4)
    validated = interactive.validate_interactive_frame(frame, palette)

    assert tuple(validated["core_number"].drop_duplicates()) == SO2_CORE_NUMBERS
    assert sorted(set(validated["contextual_direct_cluster"])) == [
        "D0",
        "D1",
        "D2",
        "D3",
    ]

    incomplete = frame.loc[frame["core_number"] != SO2_CORE_NUMBERS[-1]]
    with pytest.raises(ValueError, match="14 SO2 cores|locked order"):
        interactive.validate_interactive_frame(incomplete, palette)

    wrong_namespace = frame.copy()
    wrong_namespace["contextual_direct_cluster"] = wrong_namespace[
        "contextual_direct_cluster"
    ].str.replace("D", "C", regex=False)
    with pytest.raises(ValueError, match="D<number>|D0"):
        interactive.validate_interactive_frame(wrong_namespace, palette)


def test_payload_round_trips_every_point_without_identifiers() -> None:
    frame = _frame(cluster_count=6)
    palette = _palette(cluster_count=6)
    payload = interactive.build_interactive_payload(frame, palette)

    assert payload["method"] == "direct_hl_cosine_knn_leiden"
    assert payload["clusters"] == [f"D{index}" for index in range(6)]
    assert payload["core_order"] == list(SO2_CORE_NUMBERS)
    assert payload["point_count"] == len(frame)

    observed = 0
    for record, core_number in zip(payload["cores"], SO2_CORE_NUMBERS, strict=True):
        expected = frame.loc[frame["core_number"] == core_number]
        x = np.frombuffer(base64.b64decode(record["x_b64"]), dtype="<f8")
        y = np.frombuffer(base64.b64decode(record["y_b64"]), dtype="<f8")
        codes = np.frombuffer(
            base64.b64decode(record["cluster_codes_b64"]), dtype=np.uint8
        )
        np.testing.assert_array_equal(x, expected["x_um"].to_numpy())
        np.testing.assert_array_equal(y, expected["y_um"].to_numpy())
        np.testing.assert_array_equal(
            codes,
            expected["contextual_direct_cluster_number"].to_numpy(dtype=np.uint8),
        )
        observed += len(codes)
    assert observed == len(frame)

    serialized = json.dumps(payload, separators=(",", ":"))
    for prohibited in (
        "cell_key",
        "cell_index",
        "global_cell_index",
        "core_alias",
        "contextual_direct_cluster_number",
    ):
        assert prohibited not in serialized


def test_html_is_offline_private_dynamic_and_explicitly_no_pca(tmp_path: Path) -> None:
    frame = _frame(cluster_count=5)
    palette = _palette(cluster_count=5)
    source_table = tmp_path / "cell_contextual_direct_clusters.parquet"
    source_manifest = tmp_path / "manifest.json"
    html_path = tmp_path / interactive.HTML_FILENAME
    frame.to_parquet(source_table, index=False)
    source_manifest.write_text('{"synthetic":true}\n', encoding="utf-8")

    receipt = interactive.render_interactive_html(
        frame=frame,
        palette=palette,
        output_path=html_path,
        source_table_path=source_table,
        source_manifest_path=source_manifest,
    )
    html = html_path.read_text(encoding="utf-8")
    probe = _HTMLProbe()
    probe.feed(html)

    assert probe.canvas_cores == [str(core) for core in SO2_CORE_NUMBERS]
    assert probe.cluster_buttons == ["D0", "D1", "D2", "D3", "D4"]
    assert probe.external_assets == []
    assert probe.csp and all("connect-src 'none'" in value for value in probe.csp)
    assert "Direct 256-dimensional hL" in html
    assert "PCA and mean-centering were not used" in html
    assert "PCA-derived C clusters" in html
    for interaction in (
        "cluster-chip",
        "selectedCluster",
        "show-all",
        "Escape",
        "wheel",
        "pointerdown",
        "dblclick",
        "hover",
        "export-png",
    ):
        assert interaction in html
    for prohibited in (
        frame.loc[0, "cell_key"],
        "global_cell_index",
        "contextual_direct_cluster_number",
        "/workspace",
        "fetch(",
        "XMLHttpRequest",
    ):
        assert prohibited not in html

    assert receipt["cluster_count"] == 5
    assert receipt["pca"] is False
    assert receipt["mean_center"] is False
    assert receipt["l2_normalize_for_cosine"] is True
    assert receipt["html_sha256"] == _sha256(html_path)
    assert receipt["source_table_sha256"] == _sha256(source_table)
    assert receipt["source_manifest_sha256"] == _sha256(source_manifest)


def test_transfer_zip_is_deterministic_and_contains_only_identical_html(
    tmp_path: Path,
) -> None:
    html_path = tmp_path / interactive.HTML_FILENAME
    html_path.write_text("<!doctype html><title>offline</title>\n", encoding="utf-8")
    first_zip = tmp_path / "first.zip"
    second_zip = tmp_path / "second.zip"

    first = interactive._write_transfer_zip(html_path=html_path, zip_path=first_zip)
    second = interactive._write_transfer_zip(html_path=html_path, zip_path=second_zip)

    assert first_zip.read_bytes() == second_zip.read_bytes()
    assert first["zip_sha256"] == second["zip_sha256"]
    assert first["member_sha256"] == _sha256(html_path)
    with zipfile.ZipFile(first_zip) as archive:
        assert archive.namelist() == [html_path.name]
        assert archive.read(html_path.name) == html_path.read_bytes()


def test_final_manifest_verifies_dynamic_labels_and_detects_file_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame(cluster_count=3)
    palette = _palette(cluster_count=3)
    source_table = tmp_path / "source.parquet"
    source_manifest = tmp_path / "source.json"
    output_root = tmp_path / "viewer"
    output_root.mkdir()
    html_path = output_root / interactive.HTML_FILENAME
    zip_path = output_root / interactive.ZIP_FILENAME
    frame.to_parquet(source_table, index=False)
    source_manifest.write_text('{"synthetic":true}\n', encoding="utf-8")

    render_receipt = interactive.render_interactive_html(
        frame=frame,
        palette=palette,
        output_path=html_path,
        source_table_path=source_table,
        source_manifest_path=source_manifest,
    )
    zip_receipt = interactive._write_transfer_zip(
        html_path=html_path,
        zip_path=zip_path,
    )
    (output_root / "README.md").write_text("synthetic\n", encoding="utf-8")
    per_core_counts = {
        core_number: 3 for core_number in SO2_CORE_NUMBERS
    }
    monkeypatch.setattr(interactive, "EXPECTED_TOTAL_CELLS", len(frame))
    monkeypatch.setattr(
        interactive,
        "EXPECTED_CELL_COUNTS_BY_CORE",
        per_core_counts,
    )
    manifest = interactive._receipt_with_self_hash(
        {
            "schema": interactive.INTERACTIVE_SCHEMA,
            "status": "complete",
            "pipeline_kind": interactive.SOURCE_PIPELINE_KIND,
            "pca": False,
            "mean_center": False,
            "l2_normalize_for_cosine": True,
            "core_order": list(SO2_CORE_NUMBERS),
            "core_cell_counts": {
                str(core): count for core, count in per_core_counts.items()
            },
            "point_count": len(frame),
            "cluster_count": 3,
            "cluster_labels": ["D0", "D1", "D2"],
            "embedding_dimension": 256,
            "n_neighbors": 30,
            "leiden_resolution": 1.0,
            "render_receipt": dict(render_receipt),
            "zip_receipt": dict(zip_receipt),
            "files": interactive._file_manifest(output_root),
        }
    )

    interactive._verify_interactive_manifest(output_root, manifest)
    html_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|receipt|output"):
        interactive._verify_interactive_manifest(output_root, manifest)


def test_runner_consumes_separate_direct_report_and_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "synthetic-run"
    paths = _paths(tmp_path)
    source_root = (
        paths.report_root
        / "analyses"
        / interactive.DEFAULT_SOURCE_REPORT
        / run_id
    )
    table_path = source_root / interactive.SOURCE_TABLE_RELATIVE_PATH
    palette_path = source_root / interactive.SOURCE_PALETTE_RELATIVE_PATH
    figure_path = source_root / interactive.SOURCE_FIGURE_RELATIVE_PATH
    manifest_path = source_root / "manifest.json"
    table_path.parent.mkdir(parents=True)
    palette_path.parent.mkdir(parents=True)
    figure_path.parent.mkdir(parents=True)
    frame = _frame(cluster_count=3)
    palette = _palette(cluster_count=3)
    frame.to_parquet(table_path, index=False)
    palette_path.write_text(
        json.dumps({"colors": palette}, sort_keys=True), encoding="utf-8"
    )
    figure_path.write_bytes(b"synthetic png")
    manifest_path.write_text('{"source":"synthetic"}\n', encoding="utf-8")
    source_manifest = {
        "run_id": run_id,
        "pipeline_kind": interactive.SOURCE_PIPELINE_KIND,
        "pca": False,
        "mean_center": False,
        "l2_normalize_for_cosine": True,
        "embedding_dimension": 256,
        "n_neighbors": 30,
        "leiden_resolution": 1.0,
        "core_order": list(SO2_CORE_NUMBERS),
        "total_cells": len(frame),
        "cluster_count": 3,
    }
    monkeypatch.setattr(
        direct_clustering,
        "load_verified_direct_hl_analysis",
        lambda output_root: source_manifest,
    )
    monkeypatch.setattr(interactive, "EXPECTED_TOTAL_CELLS", len(frame))
    monkeypatch.setattr(
        interactive,
        "EXPECTED_CELL_COUNTS_BY_CORE",
        {core_number: 3 for core_number in SO2_CORE_NUMBERS},
    )

    result = interactive.run_so2_hl_direct_interactive(
        paths=paths,
        run_id=run_id,
    )
    rerun = interactive.run_so2_hl_direct_interactive(
        paths=paths,
        run_id=run_id,
    )

    assert result == rerun
    assert result["cluster_count"] == 3
    assert result["cluster_labels"] == ["D0", "D1", "D2"]
    assert result["pca"] is False and result["mean_center"] is False
    assert Path(result["static_png"]) == figure_path
    assert Path(result["html"]).is_file()
    assert Path(result["transfer_zip"]).is_file()
    final_manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert final_manifest["source_artifacts"]["static_png"] == {
        "sha256": _sha256(figure_path),
        "size_bytes": figure_path.stat().st_size,
    }

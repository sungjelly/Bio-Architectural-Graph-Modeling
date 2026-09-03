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

import spatial_benchmark.so1_raw_expression_interactive as interactive
import spatial_benchmark.so1_raw_expression_clustering as clustering
from spatial_benchmark.paths import ProjectPaths


def _frame(cluster_count: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    global_index = 0
    for core_offset, core_number in enumerate(interactive.SO1_CORE_NUMBERS):
        for cluster_number in range(cluster_count):
            rows.append(
                {
                    "global_cell_index": global_index,
                    "cell_index": cluster_number,
                    "cell_key": f"private-so1-{core_number}-{cluster_number}",
                    "core_alias": f"SO1-C{core_number:02d}",
                    "core_number": core_number,
                    "x_um": float(core_offset + cluster_number * 10 + 0.125),
                    "y_um": float(core_offset * 2 + cluster_number * 5 + 0.25),
                    "raw_library_size": 10 + cluster_number,
                    "vendor_qc_pass": cluster_number % 2 == 0,
                    "expression_cluster_number": cluster_number,
                    "expression_cluster": f"S1E{cluster_number}",
                }
            )
            global_index += 1
    return pd.DataFrame(rows)


def _palette(cluster_count: int = 5) -> dict[str, str]:
    return {
        f"S1E{index}": f"#{(0x183153 + index * 0x1D4267) % 0x1000000:06X}"
        for index in range(cluster_count)
    }


def _warning(
    *,
    cluster: str = "S1E1",
    below: int = 7,
    cluster_cells: int = 14,
) -> interactive.LowCountDepthWarning:
    return interactive.LowCountDepthWarning(
        threshold_transcripts=20,
        cohort_below_threshold_cells=below,
        flag_rule=(
            "cluster_contains_at_least_50pct_of_all_cohort_cells_below_threshold"
        ),
        flagged_clusters=(
            interactive.LowCountClusterWarning(
                cluster=cluster,
                below_threshold_cells=below,
                cluster_cells=cluster_cells,
                fraction_below_threshold=below / cluster_cells,
            ),
        ),
    )


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
        asset = {"script": "src", "link": "href", "img": "src"}.get(tag)
        if asset and values.get(asset):
            self.external_assets.append((tag, str(values[asset])))


def test_so1_frame_requires_all_cores_and_independent_contiguous_labels() -> None:
    frame = _frame(cluster_count=5)
    palette = _palette(cluster_count=5)
    validated = interactive.validate_interactive_frame(frame, palette)

    assert tuple(validated["core_number"].drop_duplicates()) == (
        interactive.SO1_CORE_NUMBERS
    )
    assert sorted(set(validated["expression_cluster"])) == [
        "S1E0",
        "S1E1",
        "S1E2",
        "S1E3",
        "S1E4",
    ]

    missing_core = frame.loc[frame["core_number"] != 14]
    with pytest.raises(ValueError, match="14 SO1 cores|locked order"):
        interactive.validate_interactive_frame(missing_core, palette)

    misleading_so2 = frame.copy()
    misleading_so2["expression_cluster"] = misleading_so2[
        "expression_cluster"
    ].str.replace("S1E", "E", regex=False)
    misleading_palette = {
        label.replace("S1E", "E"): color for label, color in palette.items()
    }
    with pytest.raises(ValueError, match="S1E<number>|S1E0"):
        interactive.validate_interactive_frame(misleading_so2, misleading_palette)


def test_structured_low_count_warning_is_validated_not_rendered_as_free_text() -> None:
    source = {
        "low_count_depth_warning": {
            "threshold_transcripts": 20,
            "cohort_below_threshold_cells": 100,
            "flag_rule": (
                "cluster_contains_at_least_50pct_of_all_cohort_cells_below_threshold"
            ),
            "flagged_clusters": [
                {
                    "cluster": "S1E2",
                    "below_threshold_cells": 75,
                    "cluster_cells": 100,
                    "fraction_below_threshold": 0.75,
                    "untrusted_text": "<script>not rendered</script>",
                }
            ],
        }
    }
    warning = interactive.parse_low_count_depth_warning(
        source,
        valid_clusters=["S1E0", "S1E1", "S1E2"],
    )

    assert warning.threshold_transcripts == 20
    assert warning.flagged_clusters[0].cluster == "S1E2"
    assert warning.flagged_clusters[0].fraction_below_threshold == 0.75
    assert "untrusted_text" not in interactive._warning_record(warning)

    inconsistent = json.loads(json.dumps(source))
    inconsistent["low_count_depth_warning"]["flagged_clusters"][0][
        "fraction_below_threshold"
    ] = 0.25
    with pytest.raises(ValueError, match="inconsistent"):
        interactive.parse_low_count_depth_warning(
            inconsistent,
            valid_clusters=["S1E0", "S1E1", "S1E2"],
        )


def test_payload_round_trips_points_and_excludes_qc_identifiers_and_values() -> None:
    frame = _frame(cluster_count=4)
    payload = interactive.build_interactive_payload(frame, _palette(cluster_count=4))

    assert payload["core_order"] == list(interactive.SO1_CORE_NUMBERS)
    assert payload["clusters"] == ["S1E0", "S1E1", "S1E2", "S1E3"]
    assert payload["point_count"] == len(frame)
    observed = 0
    for record, core_number in zip(
        payload["cores"], interactive.SO1_CORE_NUMBERS, strict=True
    ):
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
            expected["expression_cluster_number"].to_numpy(dtype=np.uint8),
        )
        observed += len(codes)
    assert observed == len(frame)

    serialized = json.dumps(payload, separators=(",", ":"))
    for prohibited in (
        "cell_key",
        "cell_index",
        "global_cell_index",
        "core_alias",
        "raw_library_size",
        "vendor_qc_pass",
        "expression_cluster_number",
    ):
        assert prohibited not in serialized


def test_html_is_offline_private_so1_named_and_source_qc_derived(
    tmp_path: Path,
) -> None:
    frame = _frame(cluster_count=5)
    palette = _palette(cluster_count=5)
    source_table = tmp_path / "cell_expression_clusters.parquet"
    source_manifest = tmp_path / "manifest.json"
    output = tmp_path / interactive.HTML_FILENAME
    frame.to_parquet(source_table, index=False)
    source_manifest.write_text('{"synthetic":true}\n', encoding="utf-8")

    receipt = interactive.render_interactive_html(
        frame=frame,
        palette=palette,
        low_count_warning=_warning(),
        output_path=output,
        source_table_path=source_table,
        source_manifest_path=source_manifest,
    )
    html = output.read_text(encoding="utf-8")
    probe = _HTMLProbe()
    probe.feed(html)

    assert probe.canvas_cores == [
        str(core) for core in interactive.SO1_CORE_NUMBERS
    ]
    assert probe.cluster_buttons == ["S1E0", "S1E1", "S1E2", "S1E3", "S1E4"]
    assert probe.external_assets == []
    assert probe.csp and all("connect-src 'none'" in value for value in probe.csp)
    assert "SO1 Core 1" in html and "SO1 Core 14" in html
    assert "SO2 Core" not in html
    assert "BAGM_SO1_VIEWER" in html
    assert "BAGM_SO2_VIEWER" not in html
    assert "do not correspond to SO2 E labels" in html
    assert "S1E1: 7 of 14 cells (50.0%)" in html
    assert "All 161,596 source cells are displayed" in html
    assert "failed vendor QC" in html
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
        "raw_library_size",
        "vendor_qc_pass",
        "expression_cluster_number",
        "/workspace",
        "http://",
        "https://",
        "fetch(",
    ):
        assert prohibited not in html

    assert receipt["html_sha256"] == _sha256(output)
    assert receipt["source_table_sha256"] == _sha256(source_table)
    assert receipt["source_manifest_sha256"] == _sha256(source_manifest)
    assert receipt["vendor_qc_fields_included"] is False
    assert receipt["low_count_depth_warning"]["source_derived"] is True


def test_javascript_uses_so1_export_titles_and_full_interaction_contract() -> None:
    script = interactive.interactive_javascript()
    normalized = script.lower()
    assert "so1 core" in normalized
    assert "so2 core" not in normalized
    assert "so1_raw_expression_clusters_" in script
    assert "selected-last" in normalized
    assert "show-all" in normalized
    assert "escape" in normalized
    assert "wheel" in normalized
    assert "pointerdown" in normalized
    assert "dblclick" in normalized
    assert "mousemove" in normalized
    assert "png" in normalized
    assert script.count("const views = new Map();") == 1


def test_transfer_zip_is_deterministic_and_contains_only_identical_html(
    tmp_path: Path,
) -> None:
    html_path = tmp_path / interactive.HTML_FILENAME
    html_path.write_text("<!doctype html><title>SO1 offline</title>\n", encoding="utf-8")
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


def test_runner_uses_verified_so1_source_and_checksum_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    source_root = (
        paths.report_root
        / "analyses"
        / interactive.DEFAULT_SOURCE_REPORT
        / interactive.SOURCE_ANALYSIS_ID
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
    warning_document = {
        "threshold_transcripts": 20.0,
        "cohort_below_threshold_cells": 7,
        "flag_rule": (
            "cluster_contains_at_least_50pct_of_all_cohort_cells_below_threshold"
        ),
        "flagged_clusters": [
            {
                "cluster": "S1E1",
                "below_threshold_cells": 7,
                "cluster_cells": 14,
                "fraction_below_threshold": 0.5,
            }
        ],
    }
    source_manifest = {
        "analysis_id": interactive.SOURCE_ANALYSIS_ID,
        "analysis_scope": "classical_raw_expression_only_joint_clustering",
        "configuration": {
            "leiden_resolution": 1.0,
            "library_size_floor": 20.0,
        },
        "core_order": list(interactive.SO1_CORE_NUMBERS),
        "total_cells": len(frame),
        "cluster_count": 3,
        "below_library_size_floor_cells": 7,
        "low_count_depth_warning": warning_document,
    }
    frame.to_parquet(table_path, index=False)
    palette_path.write_text(
        json.dumps({"colors": palette}, sort_keys=True), encoding="utf-8"
    )
    figure_path.write_bytes(b"synthetic so1 png")
    manifest_path.write_text(
        json.dumps(source_manifest, sort_keys=True), encoding="utf-8"
    )
    per_core_counts = {
        core_number: 3 for core_number in interactive.SO1_CORE_NUMBERS
    }
    monkeypatch.setattr(interactive, "EXPECTED_TOTAL_CELLS", len(frame))
    monkeypatch.setattr(
        interactive,
        "EXPECTED_CELL_COUNTS_BY_CORE",
        per_core_counts,
    )
    monkeypatch.setattr(clustering, "EXPECTED_TOTAL_CELLS", len(frame))
    monkeypatch.setattr(
        clustering,
        "EXPECTED_CELL_COUNTS_BY_CORE",
        per_core_counts,
    )
    calls: list[Path] = []

    def fake_verify(
        *, output_root: str | Path, manifest: object, paths: object
    ) -> dict[str, object]:
        calls.append(Path(output_root))
        assert manifest == source_manifest
        return source_manifest

    monkeypatch.setattr(
        clustering,
        "verify_so1_raw_expression_clustering_bundle",
        fake_verify,
    )

    result = interactive.run_so1_raw_expression_interactive(paths=paths)
    rerun = interactive.run_so1_raw_expression_interactive(paths=paths)

    assert result == rerun
    assert calls == [source_root, source_root]
    assert result["cluster_labels"] == ["S1E0", "S1E1", "S1E2"]
    assert result["point_count"] == len(frame)
    assert Path(result["static_png"]) == figure_path
    assert Path(result["html"]).name.startswith("so1_")
    assert Path(result["transfer_zip"]).name.startswith("so1_")
    final_manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert final_manifest["source_artifacts"]["static_png"] == {
        "sha256": _sha256(figure_path),
        "size_bytes": figure_path.stat().st_size,
    }
    assert final_manifest["low_count_depth_warning"]["flagged_clusters"][0][
        "cluster"
    ] == "S1E1"


def test_so1_interactive_cli_defaults_and_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spatial_benchmark.cli as cli

    parser = cli.build_parser()
    arguments = parser.parse_args(["render-so1-raw-expression-interactive"])
    assert arguments.source_analysis_id is None
    assert arguments.source_report_dir is None
    assert arguments.output_dir is None
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "complete", "source_analysis_id": "synthetic"}

    monkeypatch.setattr(interactive, "run_so1_raw_expression_interactive", fake_run)
    paths = _paths(tmp_path)
    result = cli._dispatch(arguments, registry=object(), paths=paths)

    assert result == {"status": "complete", "source_analysis_id": "synthetic"}
    assert observed == {
        "paths": paths,
        "source_analysis_id": None,
        "source_report_dir": None,
        "output_dir": None,
    }

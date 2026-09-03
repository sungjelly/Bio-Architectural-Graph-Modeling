from __future__ import annotations

import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

import spatial_benchmark.so2_hl_interactive as interactive
from spatial_benchmark.so2_pooled_full_core import SO2_CORE_NUMBERS


CLUSTERS = tuple(f"C{index}" for index in range(19))


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _small_interactive_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    global_cell_index = 0
    for core_offset, core_number in enumerate(SO2_CORE_NUMBERS):
        for cluster_number, cluster in enumerate(CLUSTERS):
            rows.append(
                {
                    "global_cell_index": global_cell_index,
                    "cell_index": cluster_number,
                    "cell_key": f"SO2-C{core_number}:{cluster_number:08d}",
                    "core_alias": f"SO2-C{core_number}",
                    "core_number": core_number,
                    "x_um": float(10 * cluster_number + core_offset + 0.125),
                    "y_um": float(5 * cluster_number + 2 * core_offset + 0.25),
                    "contextual_cluster_number": cluster_number,
                    "contextual_cluster": cluster,
                }
            )
            global_cell_index += 1
    return pd.DataFrame(rows)


@pytest.fixture
def palette() -> dict[str, str]:
    return {
        cluster: f"#{(0x10203 + index * 0x0B1929) % 0x1000000:06X}"
        for index, cluster in enumerate(CLUSTERS)
    }


class _HTMLProbe(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canvas_cores: list[str | None] = []
        self.cluster_buttons: list[str | None] = []
        self.external_dom_assets: list[tuple[str, str]] = []
        self.csp_values: list[str] = []

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
        if tag == "meta" and values.get("http-equiv", "").lower() == (
            "content-security-policy"
        ):
            self.csp_values.append(str(values.get("content", "")))
        asset_attribute = {"script": "src", "link": "href", "img": "src"}.get(tag)
        if asset_attribute and values.get(asset_attribute):
            self.external_dom_assets.append((tag, str(values[asset_attribute])))


def test_validate_interactive_frame_requires_all_cores_and_one_complete_palette(
    palette: dict[str, str],
) -> None:
    frame = _small_interactive_frame()
    validated = interactive.validate_interactive_frame(frame, palette)

    assert tuple(validated["core_number"].drop_duplicates()) == SO2_CORE_NUMBERS
    assert len(validated) == len(SO2_CORE_NUMBERS) * len(palette)
    assert validated["global_cell_index"].is_unique
    assert validated["cell_key"].is_unique
    assert set(validated["contextual_cluster"]) == set(palette)

    missing_core = frame.loc[frame["core_number"] != SO2_CORE_NUMBERS[-1]]
    with pytest.raises(ValueError, match="core|Core|14"):
        interactive.validate_interactive_frame(missing_core, palette)

    incomplete_palette = dict(palette)
    incomplete_palette.pop("C18")
    with pytest.raises(ValueError, match="palette|cluster|C18"):
        interactive.validate_interactive_frame(frame, incomplete_palette)


def test_canvas_payload_round_trips_all_points_in_fixed_core_order(
    palette: dict[str, str],
) -> None:
    frame = _small_interactive_frame()
    payload = interactive.build_interactive_payload(frame, palette)

    assert payload["core_order"] == list(SO2_CORE_NUMBERS)
    assert payload["clusters"] == list(CLUSTERS)
    assert payload["palette"] == palette
    assert payload["point_count"] == len(frame)
    assert len(payload["cores"]) == len(SO2_CORE_NUMBERS)

    observed_points = 0
    for record, core_number in zip(
        payload["cores"], SO2_CORE_NUMBERS, strict=True
    ):
        expected = frame.loc[frame["core_number"] == core_number]
        x_values = np.frombuffer(
            base64.b64decode(record["x_b64"], validate=True), dtype="<f8"
        )
        y_values = np.frombuffer(
            base64.b64decode(record["y_b64"], validate=True), dtype="<f8"
        )
        cluster_codes = np.frombuffer(
            base64.b64decode(record["cluster_codes_b64"], validate=True),
            dtype=np.uint8,
        )

        assert record["core_number"] == core_number
        assert record["point_count"] == len(expected)
        np.testing.assert_array_equal(x_values, expected["x_um"].to_numpy())
        np.testing.assert_array_equal(y_values, expected["y_um"].to_numpy())
        np.testing.assert_array_equal(
            cluster_codes,
            expected["contextual_cluster_number"].to_numpy(dtype=np.uint8),
        )
        observed_points += len(x_values)

    assert observed_points == len(frame)
    serialized = json.dumps(payload, separators=(",", ":"))
    for prohibited in (
        "cell_key",
        "cell_index",
        "global_cell_index",
        "core_alias",
        "contextual_cluster_number",
    ):
        assert prohibited not in serialized


def test_javascript_supports_highlight_reset_navigation_hover_and_png_export() -> None:
    script = interactive.interactive_javascript()
    normalized = script.lower()

    assert "cluster-chip" in normalized
    assert "data-cluster" in normalized
    assert "opacity" in normalized
    assert "selectedcluster" in normalized
    assert "selected-last" in normalized
    assert "show-all" in normalized
    assert "escape" in normalized
    assert "wheel" in normalized
    assert "pointerdown" in normalized or "mousedown" in normalized
    assert "dblclick" in normalized
    assert "mousemove" in normalized
    assert "hover" in normalized
    assert "png" in normalized
    assert "export" in normalized


def test_rendered_canvas_html_is_offline_private_and_checksums_are_valid(
    tmp_path: Path,
    palette: dict[str, str],
) -> None:
    frame = _small_interactive_frame()
    source_table = tmp_path / "cell_contextual_clusters.parquet"
    source_manifest = tmp_path / "source_manifest.json"
    output_path = tmp_path / "contextual_hl_clusters_interactive.html"
    frame.to_parquet(source_table, index=False)
    source_manifest.write_text(
        json.dumps(
            {
                "schema": "synthetic_so2_hl_source_v1",
                "core_order": list(SO2_CORE_NUMBERS),
                "cluster_count": len(palette),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    receipt = dict(
        interactive.render_interactive_html(
            frame=frame,
            palette=palette,
            output_path=output_path,
            source_table_path=source_table,
            source_manifest_path=source_manifest,
        )
    )

    assert output_path.is_file()
    html = output_path.read_text(encoding="utf-8")
    probe = _HTMLProbe()
    probe.feed(html)
    assert probe.canvas_cores == [str(core) for core in SO2_CORE_NUMBERS]
    assert probe.cluster_buttons == list(CLUSTERS)
    assert probe.external_dom_assets == []
    assert probe.csp_values
    assert all("connect-src 'none'" in value.lower() for value in probe.csp_values)

    title_offsets = [html.index(f"SO2 Core {core}") for core in SO2_CORE_NUMBERS]
    assert title_offsets == sorted(title_offsets)
    assert "cluster-chip" in html
    assert "show-all" in html
    assert "export" in html.lower()
    assert "fetch(" not in html
    assert "xmlhttprequest" not in html.lower()
    assert "url(http" not in html.lower()
    assert "/workspace" not in html
    for prohibited in (
        "cell_key",
        "cell_index",
        "global_cell_index",
        "core_alias",
        "contextual_cluster_number",
    ):
        assert prohibited not in html
    assert frame.loc[0, "cell_key"] not in html

    assert receipt["html_sha256"] == _sha256(output_path)
    assert receipt["html_size_bytes"] == output_path.stat().st_size
    assert receipt["source_table_sha256"] == _sha256(source_table)
    assert receipt["source_manifest_sha256"] == _sha256(source_manifest)
    assert receipt["point_count"] == len(frame)
    assert receipt["core_order"] == list(SO2_CORE_NUMBERS)
    assert receipt["cluster_count"] == len(palette)
    assert receipt["self_contained"] is True
    content = dict(receipt)
    observed_self_hash = content.pop("manifest_content_sha256")
    assert observed_self_hash == _canonical_sha256(content)

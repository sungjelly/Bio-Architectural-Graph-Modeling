from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.same_gene_robustness import panel_log_cp10k


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/data/prepare_same_gene_robustness_variants.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "prepare_same_gene_robustness_variants_test_module", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _array_record(path: Path) -> dict[str, Any]:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _toy_base(root: Path) -> tuple[Path, dict[str, _MODULE.SlideSource]]:
    base = root / "base"
    base.mkdir(parents=True)
    genes = ["G1", "G2", "G3"]
    _write_json(base / "genes.json", genes)
    _write_json(base / "metadata_names.json", list(ALLOWED_METADATA_COLUMNS))
    sources: dict[str, _MODULE.SlideSource] = {}
    slide_manifests: dict[str, Any] = {}
    for slide_index, slide in enumerate(_MODULE.SLIDES, 1):
        slide_root = base / slide
        slide_root.mkdir()
        cells_per_fov = 20
        fov = np.repeat(np.asarray([1, 2], dtype=np.int32), cells_per_fov)
        # FOVs are interleaved in global space.  The component graph therefore
        # gains cross-FOV edges while the within-FOV graph remains nonempty.
        coordinates = np.column_stack(
            (
                np.concatenate(
                    (
                        np.arange(cells_per_fov, dtype=np.float64) * 10.0,
                        np.arange(cells_per_fov, dtype=np.float64) * 10.0 + 5.0,
                    )
                ),
                np.full(2 * cells_per_fov, slide_index * 1000.0),
            )
        )
        rows = np.arange(len(fov), dtype=np.int32)
        counts = np.column_stack(
            (
                rows % 7,
                (rows * 2 + slide_index) % 9,
                (rows * 3 + 2) % 11,
            )
        ).astype(np.int32)
        expression = np.log1p(counts).astype(np.float32)
        cp10k, _ = panel_log_cp10k(expression)
        metadata = np.zeros((len(fov), len(ALLOWED_METADATA_COLUMNS)), dtype=np.float32)
        metadata[:, 0] = rows
        qc = (rows % 9) != 0
        group = np.full(len(fov), slide_index * 100 + 1, dtype=np.int16)
        fold = np.full(len(fov), slide_index - 1, dtype=np.int8)
        base_eligible = np.ones(len(fov), dtype=bool)
        arrays = {
            "expression_log1p": expression,
            "metadata": metadata,
            "coordinates_um": coordinates.astype(np.float32),
            "fov": fov,
            "geometry_group": group,
            "fold": fold,
            "qc_passed": qc,
            "matched_eligible": base_eligible,
        }
        for name, value in arrays.items():
            np.save(slide_root / f"{name}.npy", value, allow_pickle=False)
        inventory = {
            name: _array_record(slide_root / f"{name}.npy") for name in arrays
        }
        slide_manifests[slide] = {"arrays": inventory}
        labels = np.asarray(
            [chr(ord("a") + (index % 12)) for index in range(len(fov))], dtype=np.str_
        )
        sources[slide] = _MODULE.SlideSource(
            expression_log1p=expression,
            expression_cp10k=cp10k,
            raw_counts=counts,
            metadata=metadata,
            coordinates_um=coordinates,
            fov=fov,
            geometry_group=group,
            fold=fold,
            qc_passed=qc,
            base_matched_eligible=base_eligible,
            cell_type_labels=labels,
            panel_log_total=np.log1p(counts.sum(axis=1, dtype=np.float64)).astype(
                np.float32
            ),
            audit={
                "maximum_raw_count_roundtrip_error": float(
                    np.max(np.abs(np.expm1(expression.astype(np.float64)) - counts))
                ),
                "roundtrip_tolerance": 2e-3,
                "base_float32_to_raw_float64_coordinate_max_abs_error_um": float(
                    np.max(np.abs(coordinates.astype(np.float32).astype(np.float64) - coordinates))
                ),
                "cp10k": {"zero_panel_total_cells": int(np.sum(counts.sum(axis=1) == 0))},
            },
        )
    processed = canonical_sha256(
        {slide: slide_manifests[slide]["arrays"] for slide in _MODULE.SLIDES}
    )
    manifest = {
        "manifest_schema_version": 1,
        "raw_snapshot": {"fingerprint": "1" * 64},
        "processed_fingerprint": processed,
        "split_fingerprint": "2" * 64,
        "gene_order_sha256": canonical_sha256(genes),
        "metadata_names": list(ALLOWED_METADATA_COLUMNS),
        "slides": slide_manifests,
    }
    _write_json(base / "manifest.json", manifest)
    return base, sources


def _materialize(tmp_path: Path, name: str = "prepared") -> tuple[Path, Path]:
    base, sources = _toy_base(tmp_path / name)
    raw = tmp_path / name / "raw"
    raw.mkdir()
    labels = {slide: source.cell_type_labels for slide, source in sources.items()}

    def source_loader(
        raw_root: Path,
        base_root: Path,
        slide: str,
        genes: list[str],
        metadata_names: list[str],
        cell_type_labels: np.ndarray,
        *,
        expected_gene_count: int,
    ) -> _MODULE.SlideSource:
        del raw_root, base_root, genes, metadata_names, cell_type_labels, expected_gene_count
        return sources[slide]

    output = tmp_path / name / "same_gene_robustness_v1"
    result = _MODULE.prepare(
        base,
        raw,
        output,
        expected_gene_count=3,
        frozen_eligible_genes=np.ones(3, dtype=bool),
        source_loader=source_loader,
        cell_type_loader=lambda _raw, _base: labels,
    )
    assert result["verified"] is True
    return base, output


def test_atomic_overlay_is_runner_compatible_and_fully_replayable(
    tmp_path: Path,
) -> None:
    base, output = _materialize(tmp_path)
    result = _MODULE.verify_prepared(output, base_root=base)

    assert result["variant_count"] == 8
    assert {path.name for path in (output / "variants").iterdir()} == {
        item.variant_id for item in _MODULE.VARIANT_SPECS
    }
    runner_keys = {
        "variant_id",
        "preprocessing_version",
        "raw_snapshot",
        "processed_fingerprint",
        "split_fingerprint",
        "variant_spec",
        "base_prepared_manifest_sha256",
    }
    for spec in _MODULE.VARIANT_SPECS:
        variant = output / "variants" / spec.variant_id
        manifest = json.loads((variant / "manifest.json").read_text(encoding="utf-8"))
        assert set(manifest) == runner_keys
        assert manifest["variant_id"] == spec.manifest_variant_id
        assert manifest["variant_spec"]["primary_eligibility_file"] == "eligible_primary.npy"
        for slide in _MODULE.SLIDES:
            expression = variant / slide / "expression_log1p.npy"
            assert expression.is_symlink()
            assert not Path(os.readlink(expression)).is_absolute()
            assert expression.resolve().is_relative_to(output.resolve())
            assert np.load(variant / slide / "coordinates_um.npy").dtype == np.float64
            assert np.array_equal(
                np.load(variant / slide / "near_degree.npy"),
                np.load(variant / slide / "permuted_near_degree.npy"),
            )

    v1_integrity = json.loads(
        (
            output
            / "variants/v1_component_log1p_all/integrity_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        int(v1_integrity["slide_graph_audits"][slide]["near_cross_fov_edge_count"])
        > 0
        for slide in _MODULE.SLIDES
    )
    v4 = json.loads(
        (output / "variants/v4_train_only_library_residual/manifest.json").read_text()
    )["variant_spec"]["residualization"]
    assert v4["kind"] == "library"
    assert v4["panel_log_total_file"] == "panel_log_total.npy"
    v6_root = output / "variants/v6_train_only_cell_type_library_residual"
    v6 = json.loads((v6_root / "manifest.json").read_text())["variant_spec"][
        "residualization"
    ]
    assert v6["kind"] == "cell_type_library"
    assert v6["cell_type_code_file"] == "cell_type_code.npy"
    assert np.load(
        v6_root / "SO_1/neighbor_near_cell_type_proportions.npy"
    ).shape == (40, 12)
    assert result["processed_fingerprint"] == json.loads(
        (output / "manifest.json").read_text()
    )["processed_fingerprint"]


def test_verify_only_fails_closed_for_content_and_symlink_tampering(
    tmp_path: Path,
) -> None:
    base, output = _materialize(tmp_path)
    payload = output / "variants/v1_component_log1p_all/SO_1/near_degree.npy"
    data = bytearray(payload.read_bytes())
    data[-1] ^= 1
    payload.write_bytes(data)
    with pytest.raises(_MODULE.RobustPreparedError, match="hash changed"):
        _MODULE.verify_prepared(output, base_root=base)

    base_two, output_two = _materialize(tmp_path, "symlink_case")
    link = output_two / "variants/v0_within_fov_log1p_all/SO_1/metadata.npy"
    link.unlink()
    link.symlink_to("/tmp/outside.npy")
    with pytest.raises(
        _MODULE.RobustPreparedError,
        match="manifest hash changed|symlink|undeclared or missing",
    ):
        _MODULE.verify_prepared(output_two, base_root=base_two)


def test_publish_refuses_existing_destination_and_fingerprint_ignores_timestamp(
    tmp_path: Path,
) -> None:
    base, output = _materialize(tmp_path, "first")
    first = json.loads((output / "manifest.json").read_text())["processed_fingerprint"]
    with pytest.raises(FileExistsError):
        _MODULE.prepare(base, tmp_path / "first/raw", output, expected_gene_count=3)

    _, replay = _materialize(tmp_path, "replay")
    second = json.loads((replay / "manifest.json").read_text())["processed_fingerprint"]
    assert first == second
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


def test_enriched_cell_type_loader_rejects_duplicate_missing_and_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    paths: dict[str, Path] = {}
    for slide_index, slide in enumerate(_MODULE.SLIDES):
        slide_root = base / slide
        slide_root.mkdir()
        np.save(slide_root / "fov.npy", np.ones(12, dtype=np.int32))
        path = tmp_path / f"{slide}.csv"
        pd.DataFrame(
            {
                "fov": np.ones(12, dtype=np.int32),
                "cell_ID": np.arange(1, 13, dtype=np.int32),
                _MODULE.CELL_TYPE_FIELD: [chr(ord("a") + i) for i in range(12)],
            }
        ).to_csv(path, index=False)
        paths[slide] = path
    monkeypatch.setattr(
        _MODULE,
        "discover_slide_raw_path",
        lambda _raw, slide, _kind: paths[slide],
    )
    loaded = _MODULE._load_cell_type_labels(tmp_path, base)
    assert len(set(np.concatenate(tuple(loaded.values())).tolist())) == 12

    frame = pd.read_csv(paths["SO_1"])
    frame.loc[1, "cell_ID"] = frame.loc[0, "cell_ID"]
    frame.to_csv(paths["SO_1"], index=False)
    with pytest.raises(_MODULE.RobustPreparedError, match="duplicate"):
        _MODULE._load_cell_type_labels(tmp_path, base)

    frame.loc[1, "cell_ID"] = 2
    frame.loc[1, _MODULE.CELL_TYPE_FIELD] = "unknown"
    frame.to_csv(paths["SO_1"], index=False)
    with pytest.raises(_MODULE.RobustPreparedError, match="unknown"):
        _MODULE._load_cell_type_labels(tmp_path, base)

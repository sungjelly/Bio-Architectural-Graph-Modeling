"""Read-only V2 audit: verify renderer path provenance before content comparison.

The original numeric auditor is preserved unchanged. This adapter verifies
every path-qualified renderer record, rebasing only analysis-local paths after
publication, and supplies content-only records to the original comparisons.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

from spatial_benchmark.paths import current_paths


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _relative(value):
    value = Path(value)
    require(not value.is_absolute() and ".." not in value.parts, "Unsafe relative figure path")
    return value


def adapt_figure_manifest(manifest, out, *, pca_root, renderer, polygon_paths, color_key_path):
    """Validate paths/checksums and strip only the validated record path fields.

    Analysis-local provenance paths remain rooted at the original staging
    directory. Their relative paths are checked before the current analysis
    directory is substituted for reading, so a moved immutable bundle works
    without rewriting its manifest. External source paths are never rebased.
    """
    out, pca_root, renderer = Path(out).resolve(), Path(pca_root).resolve(), Path(renderer).resolve()
    adapted = copy.deepcopy(manifest)
    inputs = adapted["inputs"]
    original_cells = Path(inputs["cells"]["path"])
    require(original_cells.is_absolute() and original_cells.name == "cells.parquet"
            and ".." not in original_cells.parts, "Invalid original cells provenance path")
    original_root = original_cells.parent
    require(original_root.name == out.name, "Publication changed the immutable evaluation directory identity")
    checked = []

    def verify(record, current_path, provenance_path, category):
        require(isinstance(record, dict) and set(record) == {"path", "sha256", "size_bytes"},
                f"{category}: unexpected renderer record schema")
        require(Path(record["path"]).is_absolute() and ".." not in Path(record["path"]).parts
                and Path(record["path"]) == Path(provenance_path),
                f"{category}: recorded path does not identify its expected artifact")
        current_path = Path(current_path)
        require(current_path.is_file() and current_path.stat().st_size == record["size_bytes"]
                and sha(current_path) == record["sha256"], f"{category}: file size or checksum mismatch")
        checked.append({"category": category, "recorded_path": record["path"],
                        "verified_path": str(current_path.resolve()), "sha256": record["sha256"]})
        # Path is removed only after independently checking its identity and data.
        return {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}

    for name, relative in {"cells": "cells.parquet", "h0_labels": "clustering/h0_labels.npy",
                           "hL_labels": "clustering/hL_labels.npy", "clustering_summary": "clustering/summary.json"}.items():
        inputs[name] = verify(inputs[name], out / relative, original_root / relative, f"analysis_input:{name}")
    require(Path(inputs["pca_run_dir"]) == pca_root, "Unexpected PCA source directory")
    inputs["pca_checksums"] = verify(inputs["pca_checksums"], pca_root / "checksums.json",
                                      pca_root / "checksums.json", "pca_checksums")
    for relative, record in list(inputs["pca_sources"].items()):
        path = pca_root / _relative(relative)
        inputs["pca_sources"][relative] = verify(record, path, path, f"pca_source:{relative}")
    require(set(inputs["raw_polygons"]) == set(polygon_paths), "Raw polygon slide set differs")
    for slide, record in list(inputs["raw_polygons"].items()):
        path = Path(polygon_paths[slide]).resolve()
        inputs["raw_polygons"][slide] = verify(record, path, path, f"raw_polygons:{slide}")
    inputs["renderer"] = verify(inputs["renderer"], renderer, renderer, "renderer_source")
    if inputs["historical_color_keys"] is not None:
        path = Path(color_key_path).resolve()
        inputs["historical_color_keys"] = verify(inputs["historical_color_keys"], path, path, "historical_color_keys")
    for relative, record in list(adapted["files"].items()):
        relative_path = _relative(relative)
        adapted["files"][relative] = verify(record, out / relative_path, original_root / relative_path,
                                             f"rendered_file:{relative}")
    return adapted, {"status": "passed", "original_output_root": str(original_root),
                     "current_output_root": str(out), "output_paths_rebased": original_root != out,
                     "external_paths_rebased": False, "verified_record_count": len(checked),
                     "verified_records": checked, "manifest_mutated": False}


def _polygon_paths(paths):
    result = {}
    for slide in (1, 2):
        filename = f"26040302SO_{slide}-polygons.csv"
        path = paths.data_root / "raw" / filename
        if not path.is_file():
            path = path / filename
        require(path.is_file(), f"Raw segmentation source missing for SO_{slide}")
        result[f"SO_{slide}"] = path.resolve()
    return result


def audit(out: Path) -> dict:
    out = Path(out).resolve()
    original_path = Path(__file__).with_name("audit_so12_hidden_leiden.py")
    # Each call gets a private module instance. No shared module or on-disk
    # source is patched, and all original numeric algorithms remain unchanged.
    specification = importlib.util.spec_from_file_location("_so12_numeric_audit_v1_private", original_path)
    require(specification is not None and specification.loader is not None, "Cannot load preserved numeric auditor")
    original = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(original)
    original_read = original.read
    paths = current_paths()
    config = original_read(out / "config.resolved.json")
    pca_root = paths.artifact_root / "runs/2026/10" / config["pca_run_id"]
    renderer_audits = []

    def adapted_read(path):
        value = original_read(path)
        if Path(path).resolve() == out / "report/figure_manifest.json":
            value, receipt = adapt_figure_manifest(value, out, pca_root=pca_root,
                renderer=paths.project_root / "scripts/analysis/so12_hidden_maps.py",
                polygon_paths=_polygon_paths(paths),
                color_key_path=paths.report_root / "analyses/so12_embeddings_leiden/cluster_color_keys.csv")
            renderer_audits.append(receipt)
        return value

    original.read = adapted_read
    result = original.audit(out)
    if result["comparison"]["rendering_checked"]:
        require(len(renderer_audits) == 1, "Renderer manifest was not independently audited exactly once")
    result["schema"] = "so12_hidden_leiden_independent_numeric_audit_v2"
    result["original_numeric_audit_source_sha256"] = sha(original_path)
    result["audit_source_sha256"] = sha(Path(__file__))
    result["renderer_path_provenance"] = renderer_audits[0] if renderer_audits else {"status": "not_ready"}
    result["adaptation"] = "Verified renderer path-qualified records before passing content hashes to unchanged V1 comparisons"
    return result


def smoke() -> dict:
    """Check real file hashes, preserved staging paths, and three corruptions."""
    with tempfile.TemporaryDirectory(prefix="so12-audit-v2-") as temporary:
        base = Path(temporary)
        out = base / "scratch" / "ev_fixture"
        pca = base / "pca"
        renderer = base / "source/renderer.py"
        color = base / "colors.csv"
        polygons = {"SO_1": base / "raw/slide1.csv", "SO_2": base / "raw/slide2.csv"}

        def create(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            return {"path": str(path.resolve()), "sha256": sha(path), "size_bytes": path.stat().st_size}

        manifest = {"inputs": {
            "cells": create(out / "cells.parquet", b"cells"),
            "h0_labels": create(out / "clustering/h0_labels.npy", b"h0"),
            "hL_labels": create(out / "clustering/hL_labels.npy", b"hL"),
            "clustering_summary": create(out / "clustering/summary.json", b"summary"),
            "pca_run_dir": str(pca.resolve()),
            "pca_checksums": create(pca / "checksums.json", b"checksums"),
            "pca_sources": {"methods/pca/clusters.csv.gz": create(pca / "methods/pca/clusters.csv.gz", b"pca labels")},
            "raw_polygons": {slide: create(path, slide.encode()) for slide, path in polygons.items()},
            "renderer": create(renderer, b"renderer"),
            "historical_color_keys": create(color, b"colors")},
            "files": {"figures/example.png": create(out / "figures/example.png", b"figure")}}
        kwargs = dict(pca_root=pca, renderer=renderer, polygon_paths=polygons, color_key_path=color)
        before = copy.deepcopy(manifest)
        adapted, first = adapt_figure_manifest(manifest, out, **kwargs)
        require(manifest == before and not first["output_paths_rebased"], "In-place adapter changed provenance")
        require("path" not in adapted["files"]["figures/example.png"], "Validated record was not adapted")
        moved = base / "published" / out.name
        moved.parent.mkdir()
        out.rename(moved)
        moved_adapted, second = adapt_figure_manifest(manifest, moved, **kwargs)
        require(second["output_paths_rebased"] and not second["external_paths_rebased"]
                and moved_adapted == adapted and manifest == before, "Moved-bundle provenance adaptation failed")
        corruptions = []
        bad_path = copy.deepcopy(manifest)
        bad_path["files"]["figures/example.png"]["path"] = str(out / "figures/wrong.png")
        corruptions.append(bad_path)
        bad_hash = copy.deepcopy(manifest)
        bad_hash["inputs"]["raw_polygons"]["SO_1"]["sha256"] = "0" * 64
        corruptions.append(bad_hash)
        bad_external = copy.deepcopy(manifest)
        bad_external["inputs"]["renderer"]["path"] = str(base / "other_renderer.py")
        corruptions.append(bad_external)
        for bad in corruptions:
            try:
                adapt_figure_manifest(bad, moved, **kwargs)
            except ValueError:
                continue
            raise AssertionError("Corrupt renderer provenance was accepted")
    return {"status": "passed", "in_place": True, "moved_bundle": True,
            "input_manifest_unchanged": True, "corruptions_rejected": len(corruptions),
            "verified_records_per_fixture": first["verified_record_count"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and args.output is None:
        parser.error("--output is required unless --smoke is supplied")
    print(json.dumps(smoke() if args.smoke else audit(args.output), indent=2, allow_nan=False), flush=True)

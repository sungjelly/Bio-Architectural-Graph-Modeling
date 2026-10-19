"""Membership, frozen-transform, and physically absent test-array checks."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS
from spatial_benchmark.paths import current_paths
from spatial_benchmark.relative_geometry import _array_sha256
from spatial_benchmark.so12_relative_qkv_data import fit_training_metadata
from spatial_benchmark import so12_paper_split_data as data


def test_approved_partition_is_complete_disjoint_and_exact():
    assert data.VALIDATION_CORE_NUMBERS == (2, 5, 13, 27, 28)
    assert data.TEST_CORE_NUMBERS == (15, 23)
    roles = [set(values) for values in data.ROLE_CORES.values()]
    assert set.union(*roles) == set(range(1, 29))
    assert sum(map(len, roles)) == 28
    assert len(data.TRAIN_CORE_NUMBERS) == 21
    assert data.ROLE_CELLS == {"train": 264491, "validation": 61561, "test": 81607}


def test_fit_never_consumes_holdout_arrays_and_matches_independent_formula():
    train = {1: np.array([[1, 4, 2], [3, 2, 8]], np.int32),
             3: np.array([[9, 1, 6], [5, 2, 1]], np.int32)}
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Held-out array was consumed")
    supplied = {**train, 2: Forbidden(), 15: Forbidden(), 23: Forbidden()}
    actual = data.fit_training_pearson(supplied, (1, 3))
    values = np.concatenate(list(train.values())).astype(np.float64)
    p = (values.sum(0) / values.sum()).astype(np.float32)
    mu = values.sum(1)[:, None] * p.astype(np.float64)
    residuals = (values - mu) / np.sqrt(1.01 * mu)
    np.testing.assert_array_equal(actual["gene_frequency"], p)
    np.testing.assert_allclose(actual["mean"], residuals.mean(0), rtol=1e-7)
    np.testing.assert_allclose(actual["sd"], residuals.std(0, ddof=1), rtol=1e-7)
    expected = np.minimum((residuals - actual["mean"]) / actual["sd"], 10).astype(np.float32)
    np.testing.assert_array_equal(data.pearson_targets(values.astype(np.int32), p,
                                 actual["mean"], actual["sd"]), expected)


def test_frozen_morphology_matches_original_train_only_transform():
    raw = np.arange(6 * 22, dtype=float).reshape(6, 22) + 1
    flags = np.array([True, True, True, True, False, False])
    expected, stats = fit_training_metadata(raw, flags)
    np.testing.assert_array_equal(data.apply_frozen_metadata(raw, stats), expected)
    changed = raw.copy()
    changed[~flags] *= 1000
    np.testing.assert_array_equal(data.apply_frozen_metadata(changed, stats)[flags], expected[flags])


@pytest.mark.parametrize("roles", [("train", "test"), ("validation", "test"),
                                    ("train", "train"), (), "train"])
def test_invalid_role_selections_fail_before_filesystem_access(roles):
    with pytest.raises(data.SO12DataContractError):
        data.load_so12_paper_data(roles=roles)


@pytest.fixture
def small_sources(tmp_path, monkeypatch):
    """Real NPZ/graph files for 26 cores; test files deliberately do not exist."""
    paths = current_paths({"BAGM_ROOT": str(tmp_path), "BAGM_DATA_ROOT": "alternate_data"})
    monkeypatch.setattr(data, "N_GENES", 3)
    monkeypatch.setattr(data, "ROLE_CELLS", {role: len(cores) * 2 for role, cores in data.ROLE_CORES.items()})
    monkeypatch.setattr(data, "EXPECTED_CELLS", 56)
    monkeypatch.setattr(data.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    context = {"genes": ["A", "B", "C"], "cohorts": {}, "graphs": {}, "sources": []}
    observations = []
    deferred = []
    for slide in (1, 2):
        root = paths.data_root / f"processed/so{slide}_14core_relative_qkv_v1"
        (root / "cores").mkdir(parents=True)
        cohort = {"cores": [], "files": {}}
        graphs = {"cores": []}
        metadata = []
        for number in range(1 if slide == 1 else 15, 15 if slide == 1 else 29):
            alias = data.core_alias(number)
            counts = np.array([[number + 1, 2, number % 3 + 3], [4, number + 2, 6]], np.int32)
            xy = np.array([[number * 10, 1.], [number * 10 + 1, 2.]])
            coords = xy * data.PIXEL_SIZE_UM
            for index in range(2):
                observations.append(dict(core=alias, core_number=number, slide=slide,
                    donor_group=(number + 1) // 2, fov=number, cell_ID=index + 1,
                    depth=int(counts[index].sum()), x_um=coords[index, 0], y_um=coords[index, 1]))
                metadata.append(dict(slide_ID=slide, fov=number, cell_ID=index + 1,
                    CenterX_global_px=xy[index, 0], CenterY_global_px=xy[index, 1],
                    nCount_RNA=int(counts[index].sum()),
                    **{name: float(number + index + j + 1) for j, name in enumerate(ALLOWED_METADATA_COLUMNS)}))
            raw_path = root / f"cores/{alias}.npz"
            graph_root = paths.data_root / f"processed/so{slide}_14core_relative_qkv_graphs_v1/cores/{alias}"
            def write_arrays(raw_path=raw_path, graph_root=graph_root, counts=counts, coords=coords):
                np.savez(raw_path, expression_counts=counts, coordinates_um=coords)
                graph_root.mkdir(parents=True)
                np.save(graph_root / "edge_index.npy", np.array([[0, 1], [1, 0]], np.int64))
                np.save(graph_root / "relative_geometry.npy", np.zeros((2, 70), np.float32))
                np.savez(graph_root / "orientation.npz", unused=np.zeros(2))
            write_arrays()
            cohort["files"][f"cores/{alias}.npz"] = data._sha256(raw_path)
            graph_record = {"alias": alias, "n_cells": 2,
                "graph": {"qc": {"n_nodes": 2, "n_directed_edges": 2, "self_loops": 0,
                    "cross_group_edges": 0, "receiver_major_canonical_order": True,
                    "directed_edge_pairs_are_symmetric": True},
                    "checksums": {"coordinates_um_sha256": _array_sha256("coordinates_um", coords)}},
                "files": {name: data._sha256(graph_root / name) for name in
                          ("edge_index.npy", "relative_geometry.npy", "orientation.npz")}}
            graph_record["record_sha256"] = data._canonical_sha256(graph_record)
            graphs["cores"].append(graph_record)
            cohort["cores"].append({"alias": alias, "original_core_number": number, "cell_count": 2})
            if number in data.TEST_CORE_NUMBERS:
                raw_path.unlink()
                for path in graph_root.iterdir():
                    path.unlink()
                graph_root.rmdir()
                deferred.append(write_arrays)
        metadata_path = paths.data_root / f"raw/slide{slide}_metadata.csv"
        metadata_path.parent.mkdir(exist_ok=True)
        pd.DataFrame(metadata).to_csv(metadata_path, index=False)
        for record in cohort["cores"]:
            record["source_artifacts"] = {"metadata": metadata_path.name}
            record["source_checksums"] = {"metadata_sha256": data._sha256(metadata_path)}
        context["cohorts"][slide] = cohort
        context["graphs"][slide] = graphs
    context["observations"] = pd.DataFrame(observations)
    monkeypatch.setattr(data, "_source_context", lambda paths, verify: context)
    return paths, deferred


def test_preparation_and_default_load_work_with_test_arrays_absent(small_sources):
    paths, _ = small_sources
    manifest = data.prepare_so12_paper_data(paths)
    dataset = data.load_so12_paper_data(paths=paths)
    assert len(dataset.train_cores) == 21
    assert len(dataset.validation_cores) == 5
    assert not dataset.test_cores
    assert not dataset.provenance["test_arrays_opened"]
    assert manifest["test_arrays_opened_during_preparation"] == []
    assert manifest["preprocessing"]["training_core_numbers"] == list(data.TRAIN_CORE_NUMBERS)
    assert len(manifest["matrices"]) == 26
    assert set(dataset.provenance["opened_core_aliases"]).isdisjoint({"SO2-C15", "SO2-C23"})
    assert all(core.edge_index.shape == (2, 2) for core in dataset.cores)
    assert not (paths.data_root / data.CACHE_RELATIVE_PATH / "targets/SO2-C15.npy").exists()
    assert data.prepare_so12_paper_data(paths) == manifest


def test_explicit_test_reuses_frozen_parameters_and_never_writes_targets(small_sources):
    paths, create_test = small_sources
    data.prepare_so12_paper_data(paths)
    training = data.load_so12_paper_data(paths=paths)
    cache = paths.data_root / data.CACHE_RELATIVE_PATH
    before = {str(p.relative_to(cache)): data._sha256(p) for p in cache.rglob("*") if p.is_file()}
    for write in create_test:
        write()
    testing = data.load_so12_paper_data(paths=paths, roles=("test",))
    assert len(testing.test_cores) == 2
    assert not testing.train_cores and not testing.validation_cores
    for name in ("gene_p", "residual_mean", "residual_sd", "target_mean"):
        np.testing.assert_array_equal(getattr(training, name), getattr(testing, name))
    assert testing.provenance["preprocessing_fingerprint"] == training.provenance["preprocessing_fingerprint"]
    after = {str(p.relative_to(cache)): data._sha256(p) for p in cache.rglob("*") if p.is_file()}
    assert before == after


def test_corrupted_target_and_wrong_config_are_rejected(small_sources):
    paths, _ = small_sources
    data.prepare_so12_paper_data(paths)
    with pytest.raises(data.SO12DataContractError, match="split_id"):
        data.load_so12_paper_data({"dataset": {"split_id": "old_split"}}, paths)
    target = paths.data_root / data.CACHE_RELATIVE_PATH / "targets/SO1-C01.npy"
    with target.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(data.SO12DataContractError, match="checksum mismatch"):
        data.load_so12_paper_data(paths=paths, load_graphs=False)

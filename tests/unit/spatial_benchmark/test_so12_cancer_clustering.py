import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from scripts.analysis import so12_cancer_clustering as clustering


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("BAGM_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("BAGM_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(clustering, "CORE_COUNTS", {15:32,23:32})
    output = tmp_path / "scratch/active_runs/control"
    (output / "embeddings").mkdir(parents=True)
    cores = np.repeat([15,23],32)
    cells = pd.DataFrame({"row_index":range(64),"core_number":cores,"slide":2,"fov":cores,
        "cell_ID":np.tile(np.arange(32),2),"x_um":np.arange(64)*10.,"y_um":0.,"split":"test"})
    cells.to_csv(output / "cells.csv.gz",index=False)
    planted = np.tile(np.repeat([0,1],16),2)
    values = np.random.default_rng(12).normal(0,.03,(64,256)).astype(np.float32)
    values[np.arange(64),planted] += 3
    path = output / "embeddings/h1.npy"
    np.save(path,values)
    clustering.dump(output / "layers_manifest.json",{"status":"complete","passed":True,
        "cells_sha256":clustering.sha(output / "cells.csv.gz"),
        "embeddings":{"h1":{"path":"embeddings/h1.npy","sha256":clustering.sha(path)}}})
    return output, planted, tmp_path


def test_common_graph_and_parallel_fixed_partitions_preserve_planted_groups(synthetic):
    output, planted, _ = synthetic
    graph = clustering.graph(output,"h1",device="cpu",query_chunk=11,threads=1)
    assert graph["cpu_exact_audit"]["passed"] and not graph["spatial_edges_added"]
    assert graph["weights"] == "unweighted" and graph["k"] == 30
    result = clustering.leiden(output,workers=2,methods=["h1"])
    assert len(result["partitions"]) == 8 and not result["all_methods_complete"]
    assert not (output / "clustering_summary.json").exists()
    assert (output / "clustering/partial_summary_h1.json").is_file()
    for row in result["partitions"]:
        labels = np.load(output / row["labels"]["path"])
        if row["resolution"] == 1.:
            assert adjusted_rand_score(labels,planted) == 1
        assert row["leiden"]["n_iterations"] == -1
    assert clustering.graph(output,"h1",device="cpu")["files"] == graph["files"]


def test_declared_layer_checksum_precedes_search(synthetic,monkeypatch):
    output,_,_ = synthetic
    np.save(output / "embeddings/h1.npy",np.ones((64,256),np.float32))
    monkeypatch.setattr(clustering,"exact_cosine_knn",lambda *args,**kwargs:pytest.fail("Must stop before search"))
    with pytest.raises(ValueError,match="Layer embedding/cell checksum"):
        clustering.graph(output,"h1",device="cpu")


def test_cached_graph_and_resolution_one_labels_are_verified_references(synthetic,monkeypatch):
    output,planted,tmp_path = synthetic
    graph = clustering.graph(output,"h1",device="cpu",threads=1)
    source = tmp_path / "reports/reference"
    (source / "clustering").mkdir(parents=True)
    (source / "embeddings").mkdir()
    mappings = {"cells.csv.gz":output / "cells.csv.gz","embeddings/h0.npy":output / "embeddings/h1.npy",
        "clustering/h0_knn_indices.npy":output / "clustering/h1/indices.npy",
        "clustering/h0_knn_cosine_distances.npy":output / "clustering/h1/cosine_distances.npy",
        "clustering/h0_undirected_edges.npy":output / "clustering/h1/undirected_edges.npy"}
    for name,path in mappings.items():
        shutil.copyfile(path,source / name)
    for seed in (0,1):
        np.save(source / f"clustering/h0_labels_seed{seed}.npy",planted.astype(np.int32))
    clustering.dump(source / "clustering/clustering_manifest.json",{"status":"complete","k":30,
        "normalization":"row_L2_for_cosine_only",
        "symmetrization":"undirected_union","edge_weights":"unweighted","resolution":1.,
        "primary_seed":0,"stability_seed":1,"cpu_exact_audit":{"passed":True},"search":graph["search"],
        "leiden":{"0":{"random_seed":0},"1":{"random_seed":1}}})
    files = {str(path.relative_to(source)):{"sha256":clustering.sha(path)} for path in source.rglob("*") if path.is_file()}
    clustering.dump(source / "manifest.json",{"files":files})
    clustering.dump(source / "_SUCCESS",{"manifest_sha256":clustering.sha(source / "manifest.json")})
    monkeypatch.setitem(clustering.REFERENCES,"h0",("reference","h0"))
    cached = clustering.graph(output,"h0",device="cpu")
    assert cached["reused_reference"] and not cached["files"]["edges"]["relative"]
    result = clustering._partition_task((str(output),"h0",1.,0))
    assert result["reused_resolution_one"]
    np.testing.assert_array_equal(np.load(output / result["labels"]["path"]),planted)
    assert not (output / "clustering/h0/undirected_edges.npy").exists()
    # Immutable cached arrays cannot change silently on a rerun.
    np.save(source / "clustering/h0_labels_seed1.npy",1-planted)
    with pytest.raises(ValueError,match="Artifact changed"):
        clustering.graph(output,"h0",device="cpu")

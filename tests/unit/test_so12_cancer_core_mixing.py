"""Core mixing is a descriptive count, independently checked on tiny graphs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SPEC=importlib.util.spec_from_file_location("core_mixing",Path(__file__).resolve().parents[2]/"scripts/analysis/so12_cancer_core_mixing.py")
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)


def fixture_arrays():
    return (np.array([[0,1],[1,4],[2,5],[4,5],[6,7]],np.int32),
            np.array([15]*4+[23]*4),np.array([0,0,1,1,1,1,2,2],np.int32))


def test_exact_edge_and_cluster_counts_are_distinct():
    row,composition=module.analyze(*fixture_arrays(),chunk_size=2)
    assert row["cross_core_edges"]==2 and row["cross_core_edge_fraction"]==.4
    assert row["within_core_15_edges"]==1 and row["within_core_23_edges"]==2
    assert row["core_15_cross_incident_edge_fraction"]==.5
    assert row["core_23_cross_incident_edge_fraction"]==pytest.approx(1/3)
    assert row["total_clusters"]==3 and row["shared_clusters"]==1 and row["clusters_over_90pct_one_core"]==2
    assert row["core_15_clusters_unseen_in_23"]==1 and row["core_23_clusters_unseen_in_15"]==1
    assert row["core_15_cell_fraction_unseen_in_23"]==row["core_23_fraction_in_own_core_dominated_clusters"]==.5
    assert [r["cells"] for r in composition]==[2,4,2]
    assert composition[1]["core_15_fraction"]==composition[1]["core_23_fraction"]==.5


def test_exactly_ninety_percent_is_not_greater_than_ninety():
    cores=np.array([15]*10+[23]*10)
    labels=np.array([0]*9+[1]+[0]+[1]*9,np.int32)
    edges=np.array([[0,1],[9,10],[18,19]],np.int32)
    row,_=module.analyze(edges,cores,labels)
    assert row["clusters_over_90pct_one_core"]==0
    assert row["core_15_fraction_in_any_core_dominated_clusters"]==0


def test_duplicate_edge_across_chunk_boundary_is_rejected():
    edges,cores,labels=fixture_arrays()
    edges=np.insert(edges,2,edges[1],axis=0)
    with pytest.raises(ValueError,match="duplicated"):
        module.analyze(edges,cores,labels,chunk_size=2)


def test_run_reads_receipts_and_emits_only_compact_tables(tmp_path,monkeypatch):
    scratch=tmp_path/"scratch";output=scratch/"active_runs/synthetic";output.mkdir(parents=True)
    monkeypatch.setattr(module,"current_paths",lambda:SimpleNamespace(scratch_root=scratch))
    monkeypatch.setattr(module,"CORE_COUNTS",{15:4,23:4})
    edges,cores,labels=fixture_arrays()
    pd.DataFrame({"row_index":range(8),"core_number":cores,"slide":2}).to_csv(output/"cells.csv.gz",index=False)
    np.save(output/"edges.npy",edges);np.save(output/"labels.npy",labels)
    def record(name):
        p=output/name
        return {"path":name,"relative":True,"sha256":module.sha256_file(p),"size_bytes":p.stat().st_size}
    partitions=[]
    for method in module.METHODS:
        folder=output/"clustering"/method;folder.mkdir(parents=True)
        receipt={"passed":True,"status":"complete","method":method,"n_cells":8,"cells":record("cells.csv.gz"),
            "k":30,"distance":"cosine","row_l2":True,"symmetrization":"undirected_union","weights":"unweighted",
            "spatial_edges_added":False,"edges":len(edges),"files":{"edges":record("edges.npy")}}
        (folder/"graph_receipt.json").write_text(json.dumps(receipt))
        partitions.extend({"method":method,"resolution":resolution,"seed":seed,"labels":record("labels.npy")}
                          for resolution in (.25,.5,1.,2.) for seed in (0,1))
    (output/"clustering_summary.json").write_text(json.dumps({"passed":True,"status":"complete","all_methods_complete":True,
                                                             "partitions":partitions}))
    summary=module.run(output)
    assert summary["passed"] and len(summary["rows"])==7
    assert len(pd.read_csv(output/"metrics/core_cluster_composition.csv"))==21
    assert pd.read_csv(output/"metrics/core_mixing.csv").cross_core_edge_fraction.eq(.4).all()
    for name,digest in summary["output_sha256"].items():
        assert module.sha256_file(output/name)==digest

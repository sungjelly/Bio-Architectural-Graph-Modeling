"""Descriptive core mixing of frozen test graphs and primary Leiden clusters.

This diagnostic does not select or refit any scientific setting. Core-specific
clusters may reflect tissue biology, sample effects, or both; mixing itself is
not an accuracy criterion. Undirected edges are scanned in bounded chunks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths

METHODS=("h0","h1","h2","h3","h4","pca","scvi")
CORE_COUNTS={15:38145,23:43462}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def analyze(edges,core_numbers,labels,*,chunk_size=250000):
    """Pure numeric diagnostic; cell row indexes must match the graph and labels."""
    cores,labels=np.asarray(core_numbers),np.asarray(labels)
    require(cores.ndim==1 and labels.shape==cores.shape and len(cores)>0,"Core/label vectors differ")
    require(set(np.unique(cores))=={15,23},"Expected exactly cores15 and23")
    require(labels.dtype.kind in "iu" and labels.min()>=0,"Labels must be nonnegative integers")
    require(edges.ndim==2 and edges.shape[1]==2 and edges.dtype.kind in "iu" and len(edges)>0,"Invalid edge matrix")
    require(isinstance(chunk_size,int) and chunk_size>0,"Invalid edge chunk size")
    cross=within15=within23=0
    previous=None
    for start in range(0,len(edges),chunk_size):
        block=np.asarray(edges[start:start+chunk_size])
        require(np.all(block[:,0]<block[:,1]) and block.min()>=0 and block.max()<len(cores),"Invalid undirected edge endpoints")
        # Canonical union graphs are lexicographically sorted and deduplicated.
        code=block[:,0].astype(np.int64)*len(cores)+block[:,1]
        require(np.all(code[1:]>code[:-1]) and (previous is None or code[0]>previous),"Edges are duplicated or unordered")
        previous=int(code[-1])
        left,right=cores[block[:,0]],cores[block[:,1]]
        cross+=int(np.count_nonzero(left!=right))
        within15+=int(np.count_nonzero((left==15)&(right==15)))
        within23+=int(np.count_nonzero((left==23)&(right==23)))
    n_clusters=int(labels.max())+1
    by_core={core:np.bincount(labels[cores==core],minlength=n_clusters) for core in (15,23)}
    total=by_core[15]+by_core[23]
    require(np.all(total>0),"Cluster IDs must be contiguous and populated")
    # Strictly greater than90%, with integer arithmetic to avoid rounded ties.
    dominated={core:10*by_core[core]>9*total for core in (15,23)}
    any_dominated=dominated[15]|dominated[23]
    row={"edges":len(edges),"cross_core_edges":cross,"cross_core_edge_fraction":cross/len(edges),
         "within_core_15_edges":within15,"within_core_23_edges":within23,"total_clusters":n_clusters,
         "shared_clusters":int(np.count_nonzero((by_core[15]>0)&(by_core[23]>0))),
         "clusters_over_90pct_one_core":int(any_dominated.sum()),
         "core_label_shuffle_expected_cross_edge_fraction":float(2*np.count_nonzero(cores==15)*np.count_nonzero(cores==23)/(len(cores)*(len(cores)-1)))}
    compositions=[]
    for cluster in range(n_clusters):
        compositions.append({"cluster_id":cluster,"cells":int(total[cluster]),
            "core_15_cells":int(by_core[15][cluster]),"core_23_cells":int(by_core[23][cluster]),
            "core_15_fraction":float(by_core[15][cluster]/total[cluster]),
            "core_23_fraction":float(by_core[23][cluster]/total[cluster]),
            "over_90pct_core":15 if dominated[15][cluster] else 23 if dominated[23][cluster] else None})
    for core,within in ((15,within15),(23,within23)):
        other=23 if core==15 else 15
        n=int(np.count_nonzero(cores==core))
        unseen=(by_core[core]>0)&(by_core[other]==0)
        row.update({f"core_{core}_cells":n,f"core_{core}_clusters_present":int(np.count_nonzero(by_core[core])),
            f"core_{core}_dominated_clusters":int(dominated[core].sum()),
            f"core_{core}_fraction_in_any_core_dominated_clusters":float(by_core[core][any_dominated].sum()/n),
            f"core_{core}_fraction_in_own_core_dominated_clusters":float(by_core[core][dominated[core]].sum()/n),
            f"core_{core}_fraction_in_other_core_dominated_clusters":float(by_core[core][dominated[other]].sum()/n),
            f"core_{core}_clusters_unseen_in_{other}":int(unseen.sum()),
            f"core_{core}_cell_fraction_unseen_in_{other}":float(by_core[core][unseen].sum()/n),
            f"core_{core}_cross_incident_edge_fraction":cross/(2*within+cross) if 2*within+cross else None})
    return row,compositions


def resolve_record(output,item):
    path=Path(item["path"])
    if item["relative"]:
        require(not path.is_absolute() and ".." not in path.parts,"Invalid relative source path")
        path=output/path
    else:
        require(path.is_absolute(),"External reference must be absolute")
    require(path.is_file() and sha256_file(path)==item["sha256"],"Core-mixing source checksum changed")
    require(path.stat().st_size==item["size_bytes"],"Core-mixing source size changed")
    return path


def run(output):
    paths=current_paths();output=Path(output).resolve()
    active=(paths.scratch_root/"active_runs").resolve()
    require(output.is_relative_to(active) and output!=active,"Core mixing writes require an active evaluation")
    summary_path=output/"clustering_summary.json"
    summary=json.loads(summary_path.read_text())
    require(summary.get("passed") is True and summary.get("status")=="complete" and summary.get("all_methods_complete") is True,
            "Complete seven-method clustering is required")
    expected={(method,resolution,seed) for method in METHODS for resolution in (.25,.5,1.,2.) for seed in (0,1)}
    require({(p["method"],p["resolution"],p["seed"]) for p in summary["partitions"]}==expected and len(summary["partitions"])==56,
            "Full partition coverage changed")
    cells_path=output/"cells.csv.gz";cells_sha=sha256_file(cells_path)
    cells=pd.read_csv(cells_path,usecols=["row_index","core_number","slide"])
    expected_cores=np.concatenate([np.full(count,core) for core,count in CORE_COUNTS.items()])
    require(np.array_equal(cells.row_index,np.arange(len(expected_cores))) and np.array_equal(cells.core_number,expected_cores)
            and cells.slide.eq(2).all(),"Canonical cell/core order changed")
    primary={p["method"]:p for p in summary["partitions"] if p["resolution"]==1 and p["seed"]==0}
    rows=[];composition=[];inputs={"cells.csv.gz":cells_sha,"clustering_summary.json":sha256_file(summary_path)}
    graph_sources={}
    for method in METHODS:
        receipt_path=output/f"clustering/{method}/graph_receipt.json"
        receipt=json.loads(receipt_path.read_text())
        require(receipt.get("passed") is True and receipt.get("status")=="complete" and receipt["method"]==method
            and receipt["n_cells"]==len(cells) and receipt["cells"]["sha256"]==cells_sha,"Graph identity or cell order differs")
        require(receipt["k"]==30 and receipt["distance"]=="cosine" and receipt["row_l2"] is True
            and receipt["symmetrization"]=="undirected_union" and receipt["weights"]=="unweighted"
            and receipt["spatial_edges_added"] is False,"Unexpected graph settings")
        edge_record=receipt["files"]["edges"]
        edges=np.load(resolve_record(output,edge_record),mmap_mode="r",allow_pickle=False)
        require(len(edges)==receipt["edges"],"Graph edge count differs")
        label_record=primary[method]["labels"]
        labels=np.load(resolve_record(output,label_record),mmap_mode="r",allow_pickle=False)
        row,parts=analyze(edges,cells.core_number.to_numpy(),labels)
        rows.append({"method":method,"resolution":1.,"seed":0,**row})
        composition.extend({"method":method,"resolution":1.,"seed":0,**part} for part in parts)
        inputs[str(receipt_path.relative_to(output))]=sha256_file(receipt_path)
        inputs[label_record["path"]]=label_record["sha256"]
        graph_sources[method]=edge_record
        del edges,labels
    folder=output/"metrics";folder.mkdir(exist_ok=True)
    table=folder/"core_mixing.csv";composition_path=folder/"core_cluster_composition.csv"
    pd.DataFrame(rows).to_csv(table,index=False)
    pd.DataFrame(composition).to_csv(composition_path,index=False)
    result={"schema":"so12_cancer_core_mixing_v1","status":"complete","passed":True,"n_cells":len(cells),
        "methods":list(METHODS),"resolution":1.,"seed":0,"dominance_threshold":"strictly greater than90 percent of cluster cells",
        "rows":rows,"input_sha256":inputs,"graph_sources":graph_sources,
        "output_sha256":{str(p.relative_to(output)):sha256_file(p) for p in (table,composition_path)},
        "code_sha256":sha256_file(Path(__file__)),
        "interpretation":"Descriptive core separation only. Low graph mixing or core-exclusive clusters limit cross-core union transfer but do not identify technical bias versus tissue biology. Mixing is not cancer accuracy and no setting is retuned.",
        "definitions":{"cross_core_edge_fraction":"Edges between cores divided by all unique undirected edges.",
            "cross_incident_edge_fraction":"Cross-core endpoints divided by all edge endpoints incident to the stated core.",
            "unseen_clusters":"Clusters containing cells in the stated core and zero cells in the other core; unseen cell fractions are cell-weighted, not pixel-weighted.",
            "dominance":"Cluster composition counted on all canonical cells, including cells without a scoring pixel.",
            "shuffle_expectation":"Expected cross-edge fraction under uniform core-label permutation preserving core cell counts; not a fitted null or significance test."}}
    (output/"core_mixing_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();result=run(args.output)
    print(json.dumps({"status":result["status"],"methods":result["methods"]}))

"""Small report-schema checks without real data, registries, or publication."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

DIRECTORY=Path(__file__).resolve().parents[2]/"scripts/analysis"
sys.path.insert(0,str(DIRECTORY))
SPEC=importlib.util.spec_from_file_location("so12_cancer_report_review",DIRECTORY/"so12_cancer_report.py")
reporting=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)
from so12_cancer_metrics import build_pixel_counts,evaluate_partition
import publish_so12_cancer_comparison as publication


def tiny_metrics():
    labels=np.array([0,1,0,1],np.int32)
    owner15=np.array([[0,1,-1]],np.int32)
    owner23=np.array([[2,3,-1]],np.int32)
    valid=np.ones((1,3),bool)
    counts={15:build_pixel_counts(owner15,np.array([[True,False,True]]),valid,n_cells=4),
            23:build_pixel_counts(owner23,np.array([[False,True,True]]),valid,n_cells=4)}
    evaluation=evaluate_partition(labels,counts,null_repeats=0)
    results=[{"method":method,"resolution":resolution,"seed":seed,"cluster_count":2,"evaluation":deepcopy(evaluation)}
             for method in reporting.METHODS for resolution in [.25,.5,1.,2.] for seed in [0,1]]
    return {"status":"complete","passed":True,"pixel_scores_available":True,"results":results}


def test_primary_rows_preserve_support_and_transfer_direction():
    rows=reporting.primary_rows(tiny_metrics())
    assert len(rows)==56
    assert set(rows.resolution)=={1.} and set(rows.seed)=={0}
    core15=rows.loc[(rows.method=="h4")&(rows.core==15)]
    oracle=core15.loc[(core15.estimand=="oracle")&(core15.support=="footprint")].iloc[0]
    transfer=core15.loc[(core15.estimand=="cross_core_transfer")&(core15.support=="footprint")].iloc[0]
    full=core15.loc[(core15.estimand=="oracle")&(core15.support=="full_region")].iloc[0]
    assert oracle.selected_clusters=="0" and oracle.dice_f1==1
    assert transfer.selected_clusters=="1" and transfer.dice_f1==0
    assert full.tp==1 and full.fn==1 and full.dice_f1==2/3


def test_report_writes_complete_primary_tables_and_portable_images(tmp_path,monkeypatch):
    scratch=tmp_path/"scratch";output=scratch/"active_runs/synthetic_report"
    for subdir in ("metrics","registration","figures"):
        (output/subdir).mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(reporting,"current_paths",lambda:SimpleNamespace(scratch_root=scratch))
    monkeypatch.setattr(reporting,"resolution_plot",lambda metrics,folder:Image.new("RGB",(8,8),"white").save(folder/"resolution_agreement.png"))
    monkeypatch.setattr(reporting,"union_error_maps",lambda output,metrics,registration:Image.new("RGB",(8,8),"white").save(output/"figures/oracle_union_errors.png"))
    monkeypatch.setattr(reporting,"union_error_maps",lambda folder,metrics,registration:Image.new("RGB",(8,8),"white").save(folder/"figures/oracle_union_errors.png"))
    def write(relative,payload):
        (output/relative).write_text(json.dumps(payload))
    write("metrics/pixel_agreement.json",tiny_metrics())
    write("registration/registration_manifest.json",{"passed":True,"status":"verified_approximate_section"})
    write("config.resolved.json",{"evaluation_id":"synthetic_report","source_run_id":"synthetic_source",
        "resolutions":[.25,.5,1.,2.],"fit_policy":"synthetic_train_only"})
    write("pca_manifest.json",{"passed":True,"explained_variance_ratio":[.1,.2]})
    write("scvi_manifest.json",{"passed":True,"selected_epoch_zero_based":2,"validation_elbo":123.4})
    write("core_mixing_summary.json",{"passed":True,"status":"complete"})
    pd.DataFrame([{"method":method,"cross_core_edge_fraction":.2,"total_clusters":2,
        "clusters_over_90pct_one_core":0,"core_15_cell_fraction_unseen_in_23":0.,
        "core_23_cell_fraction_unseen_in_15":0.} for method in reporting.METHODS]).to_csv(output/"metrics/core_mixing.csv",index=False)
    for name in ("layers_manifest.json","clustering_summary.json","figures/figure_manifest.json"):
        write(name,{"synthetic":True})
    for name in ("figures/combined_five_panel.png","figures/all_representations_2x7.png",
                 "registration/core_15_registration_overlay.png","registration/core_23_registration_overlay.png"):
        Image.new("RGB",(8,8),"white").save(output/name)
    summary=reporting.report(output)
    assert summary["passed"] and summary["evaluation_id"]=="synthetic_report"
    table=pd.read_csv(output/"metrics/primary_summary.csv")
    assert len(table)==7
    assert table["Oracle core 15"].eq(1).all() and table["Transfer core 15"].eq(0).all()
    detail=pd.read_csv(output/"metrics/primary_resolution1_seed0.csv")
    assert len(detail)==56 and set(detail.support)=={"footprint","full_region"}
    html=(output/"report/index.html").read_text()
    assert "data:image/png;base64," in html and "descriptive upper bound" in html
    assert "50-µm boundary sensitivity" in html
    receipt=json.loads((output/"report_manifest.json").read_text())
    assert receipt["report_sha256"]==reporting.sha256_file(output/"report/index.html")


def test_final_verifier_requires_nested_manifests_and_exact_bytes(tmp_path):
    nested=tmp_path/"provenance/reference/manifest.json"
    nested.parent.mkdir(parents=True);nested.write_text('{"synthetic":true}\n')
    files={str(nested.relative_to(tmp_path)):publication.record(nested)}
    publication.write(tmp_path/"manifest.json",{"files":files})
    publication.write(tmp_path/"_SUCCESS",{"manifest_sha256":publication.sha256_file(tmp_path/"manifest.json")})
    assert publication.verify_final(tmp_path)["files"]==files
    nested.write_text('{"synthetic":false}\n')
    with pytest.raises(RuntimeError,match="Published bytes changed"):
        publication.verify_final(tmp_path)


def test_publication_checks_current_clinical_bytes_before_claiming_unchanged(tmp_path,monkeypatch):
    output=tmp_path/"active";output.mkdir()
    clinical=tmp_path/"data/clinical";clinical.mkdir(parents=True)
    image=clinical/"Core_15.png";image.write_bytes(b"original synthetic image")
    config={"evaluation_id":publication.EVALUATION,"source_run_id":publication.RUN,
            "clinical_sources":{"Core_15.png":publication.record(image)}}
    publication.write(output/"config.resolved.json",config)
    monkeypatch.setattr(publication,"current_paths",lambda:SimpleNamespace(artifact_root=tmp_path/"artifacts",data_root=tmp_path/"data"))
    monkeypatch.setattr(publication,"verify_source",lambda *args:{})
    image.write_bytes(b"changed synthetic image")
    with pytest.raises(RuntimeError,match="Clinical source changed"):
        publication.validate(output)


def test_union_error_maps_show_exact_tp_fp_fn_and_exclude_uncovered_invalid(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    labels=np.array([0,0,1,1,0,0,1,1],np.int32)
    np.save(tmp_path/"labels.npy",labels)
    publication.write(tmp_path/"clustering_summary.json",{"partitions":[
        {"method":method,"resolution":1.,"seed":0,"labels":{"path":"labels.npy"}} for method in reporting.METHODS]})
    registration={"cores":[]}
    owner=np.array([[0,1,2],[3,-1,0]],np.int32)
    roi=np.array([[True,False,True],[False,True,True]])
    valid=np.array([[True,True,True],[True,True,False]])
    for core,offset in ((15,0),(23,4)):
        shifted=np.where(owner>=0,owner+offset,-1)
        path=tmp_path/f"core{core}.npz";np.savez(path,owner=shifted,roi=roi,valid=valid)
        registration["cores"].append({"core_number":core,"raster_path":path.name})
    # Explicit selected union isolates the raster mapping from union selection.
    evaluation={"cores":{str(core):{"oracle":{"selected_clusters":[0],"footprint":{"tp":1,"fp":1,"fn":1,"dice_f1":.5}}} for core in (15,23)}}
    metrics={"results":[{"method":method,"resolution":1.,"seed":0,"evaluation":evaluation} for method in reporting.METHODS]}
    captured=[]
    original_close=reporting.plt.close
    monkeypatch.setattr(Figure,"savefig",lambda *args,**kwargs:None)
    monkeypatch.setattr(reporting.plt,"close",lambda figure:captured.append(figure))
    reporting.union_error_maps(tmp_path,metrics,registration)
    expected=np.array([[[32,156,106],[220,70,62],[60,110,210]],
                       [[224,228,233],[255,255,255],[255,255,255]]],np.uint8)
    figure=captured[-1]
    assert len(figure.axes)==14
    for axis in figure.axes:
        assert np.array_equal(axis.images[0].get_array(),expected)
    original_close(figure)

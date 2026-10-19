"""Rendering handoff checks using synthetic canonical owners and partitions."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY=Path(__file__).resolve().parents[3]/"scripts/analysis"
sys.path.insert(0,str(DIRECTORY))
SPEC=importlib.util.spec_from_file_location("so12_cancer_figures",DIRECTORY/"so12_cancer_figures.py")
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_global_owner_maps_exact_labels_and_keeps_uncovered_white():
    owner=np.array([[-1,3],[0,2]],np.int32)
    labels=np.array([1,0,1,0])
    palette=np.array([[10,20,30],[40,50,60]],np.uint8)
    rgb=module.color_raster(owner,labels,palette)
    assert rgb[0,0].tolist()==[255,255,255]
    assert rgb[0,1].tolist()==[10,20,30]
    assert rgb[1,0].tolist()==rgb[1,1].tolist()==[40,50,60]


def test_invalid_global_owner_rejected():
    with pytest.raises(ValueError,match="canonical"):
        module.color_raster(np.array([[3]],np.int32),np.array([0,1]),np.zeros((2,3),np.uint8))


def test_primary_labels_are_fixed_r1_seed0_and_checksummed(tmp_path,monkeypatch):
    monkeypatch.setattr(module,"COUNTS",{15:2,23:2})
    records=[]
    for method in module.METHODS:
        path=tmp_path/f"{method}.npy"
        np.save(path,np.array([0,1,0,1],np.int32))
        records.append({"method":method,"resolution":1.,"seed":0,
            "labels":{"relative":True,"path":path.name,"sha256":module.sha(path)}})
    (tmp_path/"clustering_summary.json").write_text(json.dumps({"partitions":records}))
    selected=module.load_partitions(tmp_path)
    assert set(selected)==set(module.METHODS)
    np.save(tmp_path/"h0.npy",np.array([1,0,1,0],np.int32))
    with pytest.raises(ValueError,match="checksum"):
        module.load_partitions(tmp_path)


def test_missing_r1_is_not_replaced_by_another_resolution(tmp_path):
    (tmp_path/"clustering_summary.json").write_text(json.dumps({"partitions":[{"method":"h0","resolution":2.,"seed":0}]}))
    with pytest.raises(ValueError,match="Seven"):
        module.load_partitions(tmp_path)
